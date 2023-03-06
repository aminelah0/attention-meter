#################### ENV & PACKAGE ACTIONS ###################
install_env:
	@pyenv virtualenv 3.10.6 attention-env
	@pyenv local <attention-env>
	@pip --install upgrade
	@pip install ipython
	@pip install ipykernel

install_package:
	@pip install -e .

reinstall_package:
	@pip uninstall -y attention || :
	@pip install -e .
