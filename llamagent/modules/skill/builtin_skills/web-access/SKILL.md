You now have access to web fetching capabilities.

Use `web_fetch` to retrieve content from URLs.
The tool fetches the page, strips HTML tags, and returns clean text.

Guidelines:
- Provide the full URL including protocol (https://)
- Content is automatically truncated to avoid context overflow
- HTML is cleaned: script/style/nav/footer tags are removed
