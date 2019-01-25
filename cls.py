"""Simple script to clear the screen.
"""

import display1593 as display

dis = display.Display1593()
dis.connect()
dis.clear()
