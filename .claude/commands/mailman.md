---
description: View and manage the MailMan to-do list
---

Read the `mailman.json` file and display its contents.

If the user asks to:
- **Add tasks**: Update the `todo_list` section in `mailman.json` with new tasks
- **Show tasks**: Display the current `todo_list` from `mailman.json`
- **Clear tasks**: Empty the `todo_list` section
- **Hand to Claude**: Present the to-do list in an actionable format

The MailMan agent stores persistent tasks and prompts for programming work. Always preserve the existing structure and merge new items rather than overwriting.
