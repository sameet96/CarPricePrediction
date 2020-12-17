
import threading
import time


def HelloWorld():
	"""User defined Thread function"""
	print ("Hello World")
	return


def Main():
	# threads = [] # Threads list needed when we use a bulk of threads
    print( "Program started.  This program will print Hello World five times...")
    mythread = threading.Thread(target=HelloWorld)
    mythread1 = threading.Thread(target=HelloWorld)
    mythread2 = threading.Thread(target=HelloWorld)
    mythread.start()
    mythread1.start()
    mythread2.start()



if __name__ == "__main__":
	Main()