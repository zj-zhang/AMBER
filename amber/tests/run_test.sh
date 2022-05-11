coverage run -m unittest discover -p "*_test.py"
coverage report -i -m > cov_report.txt
coverage-badge -f -o coverage.svg

