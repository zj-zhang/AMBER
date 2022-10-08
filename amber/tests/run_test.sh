coverage erase
find .. -name "*.pyc" -exec rm {} \;
coverage run -m unittest discover -p "*_test.py"
coverage report -i -m > cov_report.txt
coverage-badge -f -o coverage.svg

