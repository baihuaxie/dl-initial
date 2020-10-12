### Good Principles and Practices of Programming

---

###### OOP

* some good OOP principles in general:
    * 1) it is best to access a base class's method to return a certain property, rather than that property directly
        * if the base class somehow changes the way the return is done, might also need to change how the derived classes use the property
        * so a better idea would be to rely on base class's implementation & just extend on it in the derived classes when needed
    * consequently, it is a good practice to always provide a method for each property in the base class to return its value, if the property might be used in derived classes
    * 2) use leading underscores in class names to remind that this class is an abstract base class and should not be instantiated as an object
    * 3) multiple inheritance is difficult to manage for a beginner, so if I think I need it, I might be wrong & would be better off to avoid it
        * for Python, composition would be a possible way to achieve same results of multiple inheritance



---

###### Writing comments

* [this post](https://realpython.com/python-comments-guide/) is a good synopsis on writing good comments
* avoid several pitfalls for writing code comments:
  * 1) the D.R.Y. principle: Don't Repeat Yourself
    * the opposite (bad) practice would be to write what's called "WET" comments
      * comments that basically repeats what's already obvious / can be inferred from the code itself
    * the comment should always add information to what's not explicit in the code
  * 2) not to write comment as study notes
    * put these else where in dedicated places for study notes
      * if I don't have any place, this means I should create one
    * keep in mind that all codes I write for a project is to be reviewed by some other programmer or myself in the future
  * 3) first and foremost, make my codes as clear and readable as possible in the first place
    * name my variables, functions and classes with proper names
    * design my classes and functions appropriately to not entangle too many obscure structures together
    * ...
  * 4) assume that I or someone is going to go back to look at the code after months, so write clearly about what each function or a snippet of code for complicated operations does
  * 5) avoid over-commenting
    * don't write a comment for every line of code
    * comment adds to the lines in the file; it also requires time to read
    * so only add a comment when it serves a good purpose for making the code more readable
  * 6) delete or re-write comments often as I progress through the programs
  * 7) write comments to remind myself of to-be-done parts of the code
  * 8) keep the comments short
    * a comment should be shorter than the code that it tries to support
    * if I find myself writing a long comment or even multi-line comment (if not a docstring, just a comment to one line of code), it is usually a sign that I had better refactored my code to make it clearer to begin with



---

###### Documenting codes

* check out [this post](https://realpython.com/documenting-python-code/) for a guide on docstrings and documentation of python codes
* it is good practice to write docstrings for files, classes and functions as I currently do
  * refer to the [Python docstring guidelines](https://www.python.org/dev/peps/pep-0257/#one-line-docstrings) often



---

###### Contribute to open source projects





---

###### Publish Python packages

