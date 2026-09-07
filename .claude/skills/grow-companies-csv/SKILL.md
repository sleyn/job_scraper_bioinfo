---
name: grow-companies-csv
description: Scan the BioPharmGuy biotech company directory for companies to add to config/companies.csv, probing each for a live Greenhouse/Lever/Ashby board and filtering by bioinformatics relevance before proposing additions for review.
disable-model-invocation: true
---

Grows `config/companies.csv` (the ATS-direct scrape seed list) by mining
https://biopharmguy.com/biotech-company-directory.php#geo for companies not already tracked.

This is heavy on tool calls (large directory pages, hundreds of ATS probes) — for more than
one or two regions, do this from a background fork so the raw company lists and probe noise
don't fill the main conversation.

1. **Pick regions.** Default to the bio-hub regions: Northern California, Southern California,
   Boston/New England, DC/Maryland area. Use whatever regions the user names instead, if any.
   Visit the directory's `#geo` section to find each region's current page URL — don't assume a
   URL from a past run still holds, the site's slugs can change.

2. **Fetch each region's company list.** WebFetch each region page, asking for every company
   name listed, one per line, no summarizing or truncating. Combine and dedupe the names
   (case-insensitive, across regions too) into a scratch file, one name per line.

3. **Probe for ATS + relevance.** Run:
   ```
   source .venv/bin/activate
   python scripts/detect_jobspy_ats.py --names-file <scratch-file> --check-relevance --workers 16 --delay 0.2
   ```
   This dedupes against companies already in `config/companies.csv`, guesses Greenhouse/Lever/
   Ashby board tokens, and keeps only hits with at least one job title matching
   `config/keywords.yaml`. Workday is never probed — its tenant/site slug can't be guessed from
   a name alone, so a Workday-only company here is a miss, not a rule-out.

4. **Review before merging — never auto-append.** Print the candidate rows for the user. A
   company outside core biotech (an AI lab, a CRO, a diagnostics-adjacent company) that still
   cleared the relevance filter is a judgment call, not an auto-include — ask before adding rows
   like that. Give the rest a sanity look too: this file feeds the daily scrape directly, so a
   bad row now runs every day until someone notices.

5. **Append approved rows** to `config/companies.csv` in the existing column order
   (`company_name,ats_type,board_identifier,tenant,careers_url,notes,site,wd_subdomain`), with a
   note like `verified <date> - N jobs (M relevant by title)`.

6. **Verify.** Run `python scripts/verify_companies.py` and confirm every new row prints `OK`.

7. **Never commit.** Leave the diff staged/unstaged for the user to commit themselves.
