### Description
A few sentences describing the changes proposed in this pull request.

### Status
**Ready/Work in progress/Hold**

### Types of changes
<!--- Put an `x` in all the boxes that apply, and remove the not applicable items -->
- [x] Bug fix (non-breaking change which fixes an issue)
- [ ] Breaking change (fix or new feature that would cause existing functionality to change)
- [ ] New tests added to cover the changes
- [ ] Docstrings/Documentation updated


## PR Checklist
### PR Implementer
This is a small checklist for the implementation details of this PR.

If there are any questions regarding code style or other conventions check out our 
[summary](https://github.com/gradslam/gradslam/blob/main/CONTRIBUTING.rst).

- [ ] Did you discuss the functionality or any breaking changes before?
- [ ] **Pass all tests**: did you test in local ? `pytest tests/`
- [ ] Unittests: did you add tests for your new functionality?
- [ ] Documentations: did you build documentation ? `cd docs/ && sphinx-build . _build`
- [ ] Implementation: is your code well commented and follow conventions ? `black gradslam/ && flake8 gradslam/`
- [ ] Docstrings & Typing: has your code documentation and typing ?
- [ ] Update notebooks & documentation if necessary

### gradslam Team
<details>
  <summary>gradslam team workflow</summary>
  
  - [ ] Triage
  - [ ] Assign PR to a reviewer
  - [ ] Does this PR close an Issue? (add `closes #IssueNumber` at the bottom if 
        not already in description)

</details>

### Reviewer
<details>
  <summary>Reviewer workflow</summary>

  - [ ] Do all tests pass? (Unittests, Typing, Linting, Documentation, Environment)
  - [ ] Does the implementation follow `gradslam` design conventions?
  - [ ] Is the documentation complete enough?
  - [ ] Are the tests covering simple and corner cases?
 
</details>
