#!/bin/bash

find . -type f -a -name "*.py" -exec sed -i -e 's/from funpy\./from /g' {} \;
find . -type f -a -name "*.py" -exec sed -i -e 's/import funpy\./import /g' {} \;
find . -type f -a -name "*.pyx" -exec sed -i -e 's/from funpy\./from /g' {} \;
find . -type f -a -name "*.pyx" -exec sed -i -e 's/import funpy\./import /g' {} \;
