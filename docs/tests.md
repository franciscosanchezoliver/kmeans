# How to execute tests
Go to sentinel directory and execute:
> python -m pytest

## Coverage
> python -m pytest --cov-report term tests/ --cov=kmeans/
or simply if the PYTHONPATH is already set  
> pytest --cov-report term tests/ --cov=kmeans/

### Coverage fail if it is under
pytest --cov-report term tests/ --cov=kmeans/ --cov-fail-under=80

### Generate html report in local
pytest --cov-report html:reports/coverage/cov_html tests/ --cov=kmeans/ --cov-fail-under=80
