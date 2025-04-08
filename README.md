# Dependencies
To install Pipenv:

```shell
pip install --user pipenv
```


Pipenv is used to manage dependencies. You can read more about it in the [Pipenv documentation](https://pipenv.pypa.io/en/latest/).

Install dependencies from Pipfile:

```shell 
pipenv install
```

This command creates a dedicated virtual environment isolated from the system-wide installation. The dependencies in Pipfile are installed in the environment.

### Using the environment: method 1

*** I have found personally I am already in a VENV when opening a terminal - run ```deactivate``` first before creating the pipenv env

Activate the Pipenv shell:

```shell
pipenv shell
```

It looks as though nothing has changed, but now any python commands that you run will be executed within the virtual environment:

```shell
get-command python
```

The `Source` should point to something like `C:\Users\<username>\.virtualenvs\...`, meaning that you are using a virtual environment.

To exit the environment shell use the exit command:

```shell
exit
```

### Using the environment: method 2

Alternatively, you can run all python commands using the `pipenv run` prefix:

```shell
pipenv run python
```

The command activates the virtual environment and runs the commands after the prefix.

### Adding Python packages

Any `pip` commands must now use `pipenv` instead. Check the [Pipenv documentation](https://pipenv.pypa.io/en/latest/) before running any `pip` commands.

If you needed to add packages to the project:

```shell
pipenv install <package>
```

### Retrieving Dependencies
```
pipenv run pip freeze > requirements.txt
```

to create a requirements.txt file

in colab run

!pip install -r requirements.txt

to install exact env from pipenv