# Code Review Skill

## Goal
Perform thorough, constructive code reviews.
You should use this skill whenever I ask to you to apply a review of the code prior to a pull request:


You need to do these things in order to complete the code review:

Compile a report in reviews/ and add anything that i mention.

0. Read all the modified code and understand the changes that were made. Do not make any changes in this section.

1. Read the coding guidelines and ensured any new code you add follows this. Update code to follow this if necessary. Any updated code must be tested and pass all tests. This also applies to linting, type checking and formatting. Add a section to the report about any updated code that was changed to follow the coding guidelines, including an explanation of why the change was necessary and what was changed.

2. Check for bugs, security issues, performance problems, and style improvements. Resolve this, add a section to the report about any issues found, including an explanation of the issue, its severity, and how it was addressed.

3. Look at any related code that may be impacted by the changes made and identify if:
  - a. Any existing code needs to be updated to work with the new code
  - b. Any existing code can be simplified or improved based on the new code
  - c. Any existing tests need to be updated or new tests need to be added to cover the new code and its interactions with existing code
Note the changes you've made in the report

4. Make sure to have a pass through any documentation that may be impacted by the changes and update it if necessary. Add a section to the report about any documentation that was updated, including an explanation of what was updated and why.

5. Finally, summarize the overall changes made in the code review, including any major improvements, bug fixes, or refactoring that was done. Add a section to the report summarizing the overall changes and their impact on the codebase. Add this information to the chat.

If you are unsure about any aspect of the code review, ask for clarification before proceeding. The goal is to ensure that the code is of high quality, maintainable, and follows best practices.