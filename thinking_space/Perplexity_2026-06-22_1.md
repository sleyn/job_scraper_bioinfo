<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I am searching for a new bioinformatics job. What python tools can use for automatically scrape job boards/career sites?

Yes—there are several good Python options, and for a bioinformatics job hunt I’d start with tools that can pull from company career pages and ATS-backed sites rather than only big aggregators. The most practical stack is usually **Requests + Beautiful Soup/lxml** for simple pages, **Playwright** for JavaScript-heavy sites, and **Scrapy** when you want to crawl many pages reliably.[^1][^2]

## Good Python tools

- **JobSpy**: a Python library that aggregates listings from job boards such as LinkedIn, Indeed, Glassdoor, Google, and ZipRecruiter, and returns results in a Pandas-friendly format.[^3][^4]
- **Scrapy**: best when you want a robust crawler with concurrency, retries, and pipelines for large-scale scraping.[^5][^1]
- **Playwright**: a strong choice for modern career sites that render jobs with JavaScript, and often a better default than Selenium for dynamic pages.[^6][^1]
- **Selenium**: still useful for browser automation, especially on older or awkward sites, though it is usually less elegant than Playwright for new projects.[^2][^1]
- **scrapy-playwright**: a useful hybrid if you want Scrapy’s crawling workflow but also need browser rendering for dynamic pages.[^7]
- **ATS-specific scrapers/APIs**: many company sites use Greenhouse, Lever, Ashby, or Workday, and tools exist that target those directly, which is often more reliable than scraping generic job boards.[^8][^9][^10]


## What I’d use

For a job search automation project, I’d use this approach:

1. **JobSpy** for quick aggregation across common job boards.[^4]
2. **Playwright** for company career pages that load listings dynamically.[^1][^2]
3. **Scrapy** if you want to monitor many target companies over time.[^5][^1]
4. **ATS-specific scraping** for Workday, Greenhouse, Lever, and Ashby, since many biotech and pharma employers use those platforms.[^9][^10][^8]

## Practical setup

A simple, effective pipeline is to store target companies in a CSV, scrape their career pages daily, normalize fields like title/location/date/remote status, and export to CSV or a spreadsheet for filtering. For your use case, it makes sense to prioritize keywords like “bioinformatics,” “computational biology,” “NGS,” “genomics,” “single-cell,” and “variant” in the scraper logic or downstream filtering.[^4][^7]

## One caution

Some job sites actively block scraping or have terms that restrict automated access, so it’s better to prefer official APIs, public ATS endpoints, or lightweight scraping of public career pages where allowed. For a personal job hunt, a small, targeted scraper is usually enough and much easier to maintain than a fully general one.[^8][^1][^5]

A good starting stack for you would be **JobSpy + Playwright + Pandas**, then add Scrapy only if you want scale or ongoing monitoring.[^1][^4]
<span style="display:none">[^11][^12][^13][^14][^15][^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29]</span>

<div align="center">⁂</div>

[^1]: https://scrappey.com/qa/python-web-scraping/python-scraping-libraries

[^2]: https://www.webscrapinghq.com/blog/job-posting-data-essential-extraction-techniques

[^3]: https://github.com/speedyapply/JobSpy/blob/main/pyproject.toml

[^4]: https://speedyapply-jobspy.mintlify.app/introduction

[^5]: https://www.youtube.com/watch?v=kQ0iIT2suVs

[^6]: https://www.olostep.com/blog/best-python-web-scraping-libraries

[^7]: https://rennerocha.com/posts/finding-a-job-with-web-scraping/

[^8]: https://apify.com/dami_studio/multi-ats-jobs-scraper/api/python

[^9]: https://apify.com/maverickpoint/ashby-ats-scraper/api/python

[^10]: https://apify.com/blackfalcondata/workday-scraper/api/python

[^11]: https://apify.com/novus/workable-jobs-scraper/api/python

[^12]: https://www.youtube.com/watch?v=7NqxYKzwr1Y

[^13]: https://github.com/ghiarishi/job-scraper

[^14]: https://github.com/jakubstetz/job-search-automation

[^15]: https://github.com/topics/job-board-scraper

[^16]: https://medium.com/@ppavithra22092004/building-a-job-scraper-with-python-my-web-scraping-project-1ab29158ca98

[^17]: https://www.reddit.com/r/Python/comments/15zl2ed/fastapi_job_scraper_for_linkedin_indeed/

[^18]: https://www.webscrapingapi.com/best-job-scraping-tools

[^19]: https://dev.to/swiftproxy_residential/how-to-use-web-scraping-for-job-postings-in-your-search-4e86

[^20]: https://github.com/PaulMcInnis/JobFunnel

[^21]: https://lobehub.com/mcp/lowcoordination-jobspy_mcp_server

[^22]: https://www.reddit.com/r/Python/comments/137zeq8/selenium_over_scrapy/

[^23]: https://apify.com/vnx0/lever-ats-job-scraper/api/python

[^24]: https://github.com/jharemza/workday_scraper

[^25]: https://pypi.org/project/jobspy2/

[^26]: https://www.piwheels.org/project/jobspy2/

[^27]: https://github.com/olindgallet/jobscraperv2

[^28]: https://apify.com/automation-lab/workday-jobs-scraper/api/python

[^29]: https://dev.to/franciscomoretti/automate-your-job-search-scraping-400-linkedin-jobs-with-python-47c0?context=digest

