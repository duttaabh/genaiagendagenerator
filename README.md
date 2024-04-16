#How to run this application in local PC#

1. Install Python version <= 3.10 in your PC/workspace.
2. Install a Git Client in PC/workspace.
3. Update the PATH variable to include the Python & Git executables like python, pip, git clone etc.
4. Open a terminal or a command window and type in "python" and press enter to see if you are able to successfully invoke the command.
5. Perform the same test for "git" command as well.
6. Once the tests are succesful, in the same terminal, please download the code using the "git clone <HTTP URL>" for this repository (https://github.com/duttaabh/genaiagendagenerator.git).
7. Now go to the genaiagendagenerator folder using "cd genaiagendagenerator" command in the terminal.
8. Please install the virtualenv module if you do not have it using the command "pip install virtualenv".
9. Now create a virtual environment for your application with the command "python -m venv venv".
10. We need to activate the virtual environment with the command "source venv/bin/activate".
11. Now install the required modules to run this appilcation using the command "pip install -r requirements.txt".
12. Finally run the streamlit application with the command "python -m streamlit run ui_agendagen.py"
