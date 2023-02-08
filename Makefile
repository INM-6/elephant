# This script cleans up the the elephant repository
# run `make clean` to delete:
# 	- build
#	- dist
#	- .pytest_cache
#	- elephant.egg-info

clean:
	rm -rf ./build
	rm -rf ./dist
	rm -rf ./.pytest_cache
	rm -rf ./elephant.egg-info
