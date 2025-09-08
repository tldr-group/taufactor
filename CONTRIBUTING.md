# Contributing

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

Report Bugs

Report bugs at https://github.com/tldr-group/taufactor/issues.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

## Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

## Write Documentation

TauFactor could always use more documentation, whether as part of the
official TauFactor docs, in docstrings, or even on the web in blog posts,
articles, and such.

## Submit Feedback

The best way to send feedback is to file an issue at https://github.com/tldr-group/taufactor/issues.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to implement.
-   Remember that this is a volunteer-driven project, and that contributions
    are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `taufactor` for local development.

1. Fork the `taufactor` repo on GitHub.
2. Clone your fork locally:
   ```bash
   git clone git@github.com:your_name_here/taufactor.git
   ```

3. Create a new branch for your changes.
   ```bash
   cd taufactor
   git checkout -b my_dev_branch
   ```

4. Create a python environment
   ```bash
   conda create --name myenv python=3.12
   conda activate myenv
   ```

5. Install the development dependencies and run the tests:
   ```bash
   pip install -e .[dev]
   ruff check .
   pytest
   ```

6. When you're done making changes, check that your changes the tests
   ```bash
   ruff check .
   pytest
   ```

7. Commit your changes and push the branch to your fork.
    If you added new features also provide tests to ensure their maintainance.
    ```bash
    git add my_changed_files
    git commit -m "Detailed description of changes."
    git push my_remote_fork my_dev_branch
    ```

8. Submit a pull request through the GitHub website and describe your contribution.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.

## Tips

To run a subset of tests:

pytest tests.test_taufactor

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run:

```
bump2version patch # possible: major / minor / patch
git push
git push --tags
```

Travis will then deploy to PyPI if tests pass.
