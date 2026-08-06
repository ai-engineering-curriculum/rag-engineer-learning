# Chapter 3 — Enterprise ACL Models: SharePoint, Google Drive, Salesforce

## Why this chapter exists

Chapter 2 gave you three *mechanisms* for permission-aware retrieval — query-time filter, index-time filter, per-tenant isolation. This chapter is about the *shape* your ACL layer inherits from the source systems that produce enterprise corpora. You do not get to design that shape; SharePoint / Google Drive / Salesforce designed it, and your ingestion pipeline is what turns their ACL model into whatever you put on your chunks.

Each of the three has its own quirks, its own inheritance model, its own out-of-band mechanisms, and its own "wait, that's how it works?" corner. Knowing them is the difference between a retriever that mostly-works-until-audit and one that ships.

The three systems in this chapter are chosen because between them they cover >80% of the enterprise-corpus RAG projects your track will encounter:

- **SharePoint / OneDrive / Microsoft 365** — file-and-folder ACL with inheritance and break-inheritance, using Microsoft's role-and-group model on Entra ID (formerly Azure AD).
- **Google Drive / Google Workspace** — file-and-folder ACL with a *parents* graph rather than a strict tree, plus shortcuts and shared drives.
- **Salesforce** — row-level record ACL with a stack of five compounding mechanisms (OWD, role hierarchy, sharing rules, territories, manual shares) that recompute asynchronously.

Confluence, JIRA, ServiceNow, Slack, Notion — each has its own shape, but the three above illustrate the recurring failure patterns you will meet everywhere else.

## SharePoint / OneDrive / Microsoft 365

### The model

Every securable object (site, library, folder, item) has a set of **role assignments** — pairs of `(principal, role)` where the principal is a user or an Entra ID group and the role is a permission level (Read, Contribute, Edit, Full Control, or custom). By default, an object *inherits* its role assignments from its parent — a library inherits from its site, a folder from its library, an item from its folder.

An object can **break inheritance**, at which point its role assignments become independent. New assignments can be added, existing ones removed. Re-inheriting later re-syncs to the parent but *loses* the object-specific assignments — a one-way door if you didn't record them.

Reference: [*Introduction to permissions in SharePoint*](https://learn.microsoft.com/en-us/sharepoint/introduction) and [*Understand permissions and permission inheritance in SharePoint*](https://learn.microsoft.com/en-us/sharepoint/plan-user-content-search-and-access-file-shares).

### Traps for the retriever

- **Break-inheritance items are the leak surface.** Naive ingestion reads the library ACL and applies it to every item — but a break-inheritance item has its *own* ACL. You must, per item, check whether inheritance is broken (Graph API: `hasUniqueRoleAssignments`) and if so fetch the item's own role assignments. The vast majority of items inherit; the few that don't are exactly the ones that carry compliance-sensitive content — the exec team's board deck in an otherwise-open library.
- **Group expansion is transitive and can be very deep.** A role assignment to a group means access for all members of that group, which may itself contain other groups. Entra ID lets nested groups go 10+ levels; caching effective membership per user is the only affordable way to serve queries.
- **Everyone / Everyone except external users / Authenticated Users.** These are special principals with implicit membership. Their inclusion in a role assignment means "the whole tenant can see this." An audit that flags these as "over-permissive by default" is standard practice; your retriever should surface them the same way ("this chunk is broadly accessible") for downstream monitoring.
- **External sharing.** Guests are added to a group whose membership includes users from other Entra ID tenants. Effective ACL for the retriever includes those external principals; a retriever query issued *by* an external guest must correctly resolve their principal chain.
- **Sensitivity labels are orthogonal to ACLs.** Microsoft Purview [Sensitivity labels](https://learn.microsoft.com/en-us/purview/sensitivity-labels) can apply encryption + rights-management that *further* restricts a doc — a user with read permission on the SharePoint item may still be denied by the label's own rights. If the doc is DRM-protected, your retriever shouldn't be able to read it in the first place; if the extraction pipeline decrypts it (with a service principal that has decryption rights), you've *dropped* the sensitivity label's protection when you re-serve the plaintext chunk. Document the boundary explicitly.
- **Change notification latency.** Microsoft Graph webhooks / delta queries let you react to permission changes, but the delivery is not instant — SLA is "usually within minutes." A permission revocation is not effective on your retriever until your ingestion pipeline processes the change notification and either re-embeds the item with the new ACL or evicts it.

### The ingestion / retrieval design

- Store the effective ACL per chunk as a set of Entra ID object IDs (users + groups), not names. Names get renamed; object IDs don't.
- Cache group membership per user with a short TTL (minutes). Recompute on token refresh.
- Emit a `broke_inheritance: bool` field on every chunk to make audits easy.
- Emit a `sensitivity_label: str | null` field to make the boundary to DRM-protected content visible in your logs.
- Subscribe to Graph delta queries for the sites/libraries you ingest; process the permission-change events with priority (before content-change events).

## Google Drive / Google Workspace

### The model

Every file has one or more **parents** (files can live in multiple folders — Drive is a DAG, not a tree, though the UI now discourages multi-parenting in favor of shortcuts). Every file has an ordered list of **permissions**, each of type `user`, `group`, `domain`, or `anyone`, with a role of `reader`, `commenter`, `writer`, `fileOrganizer`, `organizer`, or `owner`.

Shared Drives (formerly Team Drives) are a separate structure — files inside a Shared Drive inherit permissions from the drive itself, and per-file re-sharing is limited by the drive's sharing settings.

References: [*How Drive sharing works*](https://support.google.com/a/answer/60781), [*Overview of shared drives*](https://support.google.com/a/answer/7212025), Drive API [permissions.list](https://developers.google.com/drive/api/reference/rest/v3/permissions/list).

### Traps for the retriever

- **`anyone`-with-link is a leak.** A file with a `type: "anyone"` permission is accessible to anyone with the URL (whether inside your workspace or not). If you index such a file, your effective ACL for it is "public" — which means either every user of your retriever can see it (probably fine if the file was already meant to be world-readable) or you decline to index it (safer if you can't distinguish "public on purpose" from "public by accident"). Emit `is_public: bool` for downstream policy.
- **Domain-level permissions.** `type: "domain"` grants access to everyone in your workspace domain (or a specified domain). Your retriever must know its own tenant's domain and expand `domain` to "every user in the workspace."
- **Shortcuts.** A shortcut is a file that points to another file. The shortcut has its own permissions; the target file has its own. A user must have permission on the *target* to read it; the shortcut alone doesn't grant access. Your ingestion should follow shortcuts to the target and index the target's ACL, not the shortcut's.
- **Ownership transfer changes semantics.** Google's ownership model is single-owner-per-file (in My Drive; in Shared Drives the drive owns files). Ownership transfer changes the ACL implicitly. Your ingestion must detect these events (Changes API) and re-check ACLs on affected files.
- **Impersonation via delegated auth.** Your ingestion service account may be granted domain-wide delegation, which lets it read files on behalf of any user. This is convenient (one credential, all files) but dangerous — a bug that fetches files as `alice@` for a query issued by `bob@` is a leak. Enforce that the service account impersonates the *caller* for the retrieval path (or, safer, ingest as a dedicated ingestion identity and enforce the ACL at query time from stored ACL data).
- **File extension mismatch.** Google Docs / Sheets / Slides don't have file bytes — you export them (as `.docx`, `.xlsx`, `.pptx`, or the native `application/vnd.google-apps.*` type). The export can silently omit content that only renders in the native app (comments, suggested edits, revision history). Your chunk provenance must record which export format you used so an audit can reproduce.

### The ingestion / retrieval design

- Store the effective ACL as a set of `{type, id}` tuples — `{type: "user", id: "u@example.com"}`, `{type: "group", id: "g@example.com"}`, `{type: "domain", id: "example.com"}`, `{type: "anyone", id: null}`.
- Expand `domain` and `anyone` at query time or at index time — either works, but choose one and be consistent.
- Follow shortcuts to their targets in the ingestion pipeline; do not index the shortcut as its own document.
- Subscribe to [Drive Changes API](https://developers.google.com/drive/api/guides/changes) for delta polling. Poll `permissions.list` on any changed file.

## Salesforce

### The model

Salesforce is the one that surprises people. Instead of a file-and-folder ACL, it has *row-level* visibility on every record type, and that visibility is the composition of **five** mechanisms:

1. **Organization-Wide Defaults (OWD).** The floor. For each object type, OWD sets the default sharing model — Private, Public Read Only, Public Read/Write, or Controlled by Parent. Private is the default for anything sensitive.
2. **Role hierarchy.** Users inherit visibility upward — a manager sees everything their reports can see (if OWD is Private and the object is set to "grant access using hierarchies"). The hierarchy is a tree of roles the org configures.
3. **Sharing rules.** Criteria-based (a record with `region = 'EMEA'` shares to the EMEA group) or ownership-based (records owned by group A share to group B) rules that grant *additional* visibility on top of OWD.
4. **Territory management.** Optional. Territories are a parallel hierarchy assigning records to territories, and users to territories. Overlaps and reassignments trigger recomputes.
5. **Manual shares and Apex-managed shares.** Individual records can be shared manually or programmatically via Apex. These are per-record grants that supplement the above.

Additionally, the *sharing recalculation* runs asynchronously — a role-hierarchy change, a territory reassignment, or a bulk update can queue a sharing recompute that takes minutes to hours to complete. During that window, visibility is *stale* on the Salesforce side, so nothing your retriever does can be fresher than Salesforce itself.

References: [*Record-Level Security*](https://architect.salesforce.com/decision-guides/data-sharing), [*Sharing architecture*](https://architect.salesforce.com/fundamentals/data-security), Salesforce Developer docs on [`UserRecordAccess`](https://developer.salesforce.com/docs/atlas.en-us.object_reference.meta/object_reference/sforce_api_objects_userrecordaccess.htm).

### Traps for the retriever

- **Effective visibility is not a static ACL.** You cannot denormalize a record's read-set onto the chunk once and be done. Change the caller's role, or reassign a territory, or add a sharing rule, and the answer changes without the record's data changing.
- **The retriever must ask Salesforce whether a caller can see a record.** The correct query-time check is not "does my denormalized ACL include this user" — it is a call to [`UserRecordAccess`](https://developer.salesforce.com/docs/atlas.en-us.object_reference.meta/object_reference/sforce_api_objects_userrecordaccess.htm) for the specific `(user, record_id)` pair. Salesforce returns `HasReadAccess: true/false`. That is the authoritative answer. Anything else is an approximation.
- **`UserRecordAccess` is expensive.** One SOQL query per record is not cheap; batching helps (Salesforce accepts `WHERE RecordId IN (:ids)` up to a limit). Cache with a short TTL and be prepared to re-check on any user attribute change (role, territory, group membership).
- **Field-level security is orthogonal.** A user can see the record but not certain fields. If your embeddings encode the whole record including a field the user can't see, you've serialized a leak into the vector. Ingestion must respect FLS per field per user, which means either embedding per role class (index-time filtering, shape B) or storing per-field permissions and reconstructing the visible subset at query time. Neither is cheap; picking is a design decision.
- **The sharing recompute lag.** A user demoted at 09:00 may still be able to see previously-visible records until Salesforce finishes the recompute at 09:30. Your retriever cannot be fresher than Salesforce. Log the lag; surface it as an SLO caveat.
- **Public groups vs. queues vs. roles.** Salesforce has multiple principal types. Do not conflate them. A user's *effective* set of principals is `{user, roles, groups, queues, territories}`; every one of them can be a target of a share.
- **Portals, communities, and experience cloud users.** External users have a separate profile stack (Customer Community, Partner Community). Their visibility rules interact with sharing sets and share groups, which are yet another mechanism on top of the five above. If your corpus includes portal-visible records, read [*Sharing sets*](https://help.salesforce.com/s/articleView?id=sf.customer_portal_setting_light.htm) before assuming your model is complete.

### The ingestion / retrieval design

- **Do not denormalize the effective ACL.** Salesforce's model doesn't fit. Store the `record_id` and `object_type` on the chunk; at query time, batch-check `UserRecordAccess` for the top-N candidates against the caller and drop the ones that fail.
- Alternatively, if you *must* denormalize (query-time Salesforce round-trips are too slow), accept a shorter freshness SLA and periodically recompute per-user visibility using [Salesforce Data Cloud](https://www.salesforce.com/data/) or a warehouse mirror of the sharing tables. This is a large systems-integration project; do not underestimate it.
- Emit a `salesforce_freshness_lag_seconds` field on every response so downstream systems know how stale the check is.
- For Field-Level Security, either embed the union of visible fields per profile class (a small number of classes) and use index-time partitioning by class, or store fields on the chunk with per-field ACL and reconstruct at query time. Both are hard; the second is more correct.

## The general pattern — you inherit the ACL model, you don't design it

Every enterprise system you ingest from has *its* ACL model, and your retriever's ACL layer becomes whatever that model looks like after ingestion. The pattern:

- **Effective ACL, not source ACL.** Store the *resolved, post-inheritance, post-rule-application* set of principals that can currently read the chunk, not the source-system fields (`library_id`, `parent_folder_id`, `record_owner`) that would require you to re-resolve at query time.
- **Refresh on permission events.** ACLs change out-of-band of content. Every source system has a change-event API (Graph delta, Drive Changes, Salesforce Change Data Capture / Streaming API); subscribe to it, prioritize permission-change events, and re-index affected chunks.
- **Cache group membership per user with a short TTL.** Every ACL check needs the caller's principals; every source system requires expanding a group to its members. Cache the *caller's* memberships (small, per-user) rather than the *group's* members (large, per-group).
- **When the source system can't fit into denormalized ACLs, ask the source system.** Salesforce is the standard example; Confluence's space-vs-page inheritance can be another. A per-request round-trip to authoritative source is honest; a stale local approximation is silent-leak territory.
- **Log the effective ACL on every retrieval response.** In an incident, the log line that says "user U received chunk C, whose effective ACL is {A, B, C}, and U's principals were {A, D}" is the difference between "we can prove it wasn't a leak" and "we don't know."

## Two failure patterns to internalize

### Pattern 1 — the "we ingested the ACL wrong" incident

Ingestion pulled the parent library's ACL for every file. A subset of files had inheritance broken with a narrower ACL. Retriever returned files to users who could see the parent library but not the specific file. Nobody noticed until a security review sampled 100 responses and found 3 leaks.

**The defense:** ingest the *effective* ACL per item, not the parent's; audit sample regularly; the fuzz test from chapter 2 makes this a CI gate.

### Pattern 2 — the "we ingested the content but ignored the ACL change" incident

Ingestion runs nightly. Salesforce role hierarchy changed at 10:00 (a demotion). Nightly reingest ran at 02:00 the next day. Between 10:00 and 02:00 the retriever served the demoted user records they should not have been able to see.

**The defense:** subscribe to permission-change events and reprocess *permissions* on a much shorter cycle than *content*; for systems that don't publish a low-latency change stream, degrade gracefully by doing query-time authoritative checks against the source (accept the latency cost).

## What you should be able to do after this chapter

- Describe SharePoint's inheritance / break-inheritance model and name what your ingestion must do per-item (check `hasUniqueRoleAssignments`, fetch the item-specific role assignments, denormalize the effective ACL).
- Describe Google Drive's parents / shortcuts / shared drives model and name three ways `anyone`, `domain`, and shortcuts can trip your retriever.
- Describe Salesforce's five sharing mechanisms (OWD, role hierarchy, sharing rules, territories, manual/Apex shares) and argue why denormalizing the effective ACL to the chunk is often wrong, and what to do instead (per-record `UserRecordAccess`).
- Name the "sharing recompute lag" phenomenon and its implication for retrieval freshness SLAs.
- Author an ingestion contract that (a) stores effective ACLs, (b) subscribes to permission-change events, (c) prioritizes permission events over content events, and (d) logs the effective ACL on every retrieval response.
- Cite the boundary to `ai-risk-engineer` — this chapter tells you *what ACL to enforce*; it does not tell you how to detect PII in the extracted text before it gets embedded.

## Further reading

- Microsoft Learn: [*Overview of permissions in SharePoint*](https://learn.microsoft.com/en-us/sharepoint/introduction) and [*Get started with SharePoint sharing*](https://learn.microsoft.com/en-us/sharepoint/plan-user-content-search-and-access-file-shares) — the vendor's own docs; the break-inheritance mechanics are documented but easy to miss on first read.
- Microsoft Graph docs: [*List permissions on a driveItem*](https://learn.microsoft.com/en-us/graph/api/driveitem-list-permissions) and [*Delta query for files*](https://learn.microsoft.com/en-us/graph/delta-query-files) — the specific APIs your SharePoint ingestion uses.
- Microsoft Purview: [*Sensitivity labels overview*](https://learn.microsoft.com/en-us/purview/sensitivity-labels) — the DRM boundary that lives alongside ACLs and can further restrict content.
- Google Workspace Admin Help: [*How Drive sharing works*](https://support.google.com/a/answer/60781), [*Overview of shared drives*](https://support.google.com/a/answer/7212025).
- Google Drive API: [*Permissions*](https://developers.google.com/drive/api/reference/rest/v3/permissions) and [*Changes*](https://developers.google.com/drive/api/guides/changes) — the reference for the Drive-side APIs.
- Salesforce Architects: [*Record-Level Security*](https://architect.salesforce.com/decision-guides/data-sharing) and [*Sharing architecture*](https://architect.salesforce.com/fundamentals/data-security) — the canonical writeups of the five-mechanism model.
- Salesforce Developer docs: [*UserRecordAccess*](https://developer.salesforce.com/docs/atlas.en-us.object_reference.meta/object_reference/sforce_api_objects_userrecordaccess.htm) — the authoritative per-record visibility check.
- Salesforce Help: [*Enterprise Territory Management*](https://help.salesforce.com/s/articleView?id=sf.tm2_intro.htm) — the territory model, which most projects don't realize is orthogonal to role hierarchy until it bites them.
- Atlassian: [*Confluence — Restrict access to specific pages*](https://support.atlassian.com/confluence-cloud/docs/what-are-page-restrictions/) — Confluence's inheritance-with-per-page-restrictions shape rhymes with SharePoint's and is a useful contrast.
