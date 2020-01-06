#CIS3187 Documentation

##Chakotay Incorvaia 358199(M)


All requirements have been met in order to declare this assignment as 100% completed.

##Development Stages

###Planning Stage

Before any programming was done, a few hours were spent in understand the structure and operation of a neural network. Such as understanding the matrix multiplication required for feedforward as well as the equations required to do error back propagation.
The language chosen for this project was Python 3.8 given the wide availability of online resources as well as the prevalence of mathematical and statistical based libraries one may use.
It is important to note that no prior development was experienced with Python, and so most issues experienced in the project was due to lack of knowledge of syntax and other characteristics of the language.
Given that Python is a scripting language, the project is simply broken down into smaller bite-sized functions which perform a specific task, which in turn are called in order of operation. The entire program is run off one class called “Main.py” and was not split into numerous classes.
Finally, the project was uploaded to GitHub so that it was easier to work on the code whilst switching workstations.
Additional Note: The python version was later downgraded to Python 3.7.4 due to issues with finding an easy to use plotting and graphing library. 


The data set was placed inside the program automatically and not generated from a csv file as required. Due to only having only 5 inputs the possible numbers are only up to 32. Given that the output can only be 3, randomly generating the Boolean function is not worth the effort.
The Boolean function used in the program is of ABCDE = ¬ACD.

###Early Development

The project first started with creating the basic data structures that are to be used throughout the program. A basic data set was also created as well as the resulting target output which is passed through a Boolean function. The Boolean function is quite simply ¬ACD from the numbers 0 to 31.
Initial development was not done with a randomized data set or testing set in order to hasten production. Only one fact was used of the data set so that the feedforward can be tested.
During the early stages of development, it was noticed that calling each method individually was not good practice and so a main method was created which in turn would call the required methods. This function’s task is to encapsulate the entire program from running all the facts to calling back propagation at the appropriate time. The running of the test set and functions tasked with graphing the result would be instead run separated from the main functionality of the program.
At this stage of development, the feed forward functionality of the neural network was completed using basic matrix multiplication with the python library NumPy. Surprisingly passing an array to the sigmoid function returns a same sized array with each value within it have had sigmoid applied to it. These features greatly helped in reducing development time which would have had to be spent handling matrix multiplication.
No major issues were experienced so far early in development with regards to the functionality of the project. The biggest issues were learning the syntax of Python and issues experienced with PyCharm and git. 


###Main Development Stage

During the main stage of development, the back-propagation function was being developed. This proved to be the hardest task of the entire project as the understanding of the equations in the Planning Stage was inaccurate and needed thorough revision.
Indeed, the development for the hidden layer delta was most time consuming and once developed the neural network was not correctly fixing the graph. 
Whilst the equation for the outer layer deltas were correct, the weights were not being adjusted properly. The same issue was occurring for the hidden layer delta calculation. This issue was later found to be caused by the indices of the arrays being set poorly. The cause was because the weight adjustment equation was not set properly to use the correct layer. Adjustments to the hidden layer weights had to be set to use the inputs whilst the outer layer weights needed to make use of the hidden layer.
Another issue was found to be with the error checking, which did not work in certain cases because the function was not disregarding whether the input was negative or positive when being compared to be > 0.2. This caused the neural network to not call the backpropagation algorithm when the value was negative such as -0.3.
Once these issues were fixed, the data set was separated as instructed by shuffling the data set into a training set and reserving 20% into a separate array for the testing set.

###Final Development Stage

Once the program had its feedforward and backpropagation working successfully, it was edited in order to be able to run the entire training set and for the required number of epochs or until the condition is met.
The test set will only run once, meaning that is represents only 1 epoch which will run at the end of training.
The last stage of development required the program to plot the results of the training set and testing set on a graph. This feature was time-consuming because it was difficult in finding an appropriate library which provided enough documentation for it to be used. Numerous attempts were done with ‘plotly’ but alas they were all failures due to unclear documentation. It was then decided to downgrade python to 3.7.4 so that ‘matplotlib’ could be used instead. ‘Pandas’ was initially used for plotly but was kept for use in matplotlib by structuring a data frame from which the graph could be plotted neatly.
When the program is run, the chosen IDE should display the graph. The training set of the neural network is displayed in the line graph as a red line, whilst the test epoch is displayed as a green dot.
The y-axis represents the total percentage of bad facts in the epoch whilst the x-axis represents the number of epochs. An Observation: Whilst the training set is fully trained within the 1000 Epoch limit, the test set does not always run without a bad epoch. There are also cases in which the training set does not train itself fully within the 1000 epoch limit.
Please see below the training set results and testing results;
 
 
###Improvements Required

The following are improvements that would have been made should time have permitted.

•	Separation of main class into separate classes
•	Neater and nicer graphs
•	Latest python version
•	Proper git dependency checking by using ‘pipenv’ or another package managing tool to allow for easier setup for exam evaluation
•	Documentation included in the README of the project with images
•	Randomly generated Boolean function







