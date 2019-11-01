import time
import pyautogui

#Courtesy of: https://towardsdatascience.com/creating-ai-for-gameboy-part-1-coding-a-controller-5eb782f54ede

def press_key(key, n_times = 1):
    for _ in range(n_times):
        pyautogui.keyDown(key)
        pyautogui.keyUp(key)

def select_next_unit():
    press_key('q')
    press_key("'")

def move_unit(left=0,right=0,up=0,down=0):
    ret_list = [left,right,up,down]
    if left>right:
        press_key('a', left-right)
    elif right>left:
        press_key('d', right-left)
    if up>down:
        press_key('w', up-down)
    elif down>up:
        press_key('s', down-up)
    press_key("'")
    time.sleep(0.2)
    return ret_list

def wait():
    time.sleep(0.2)
    press_key('w')
    time.sleep(0.2)
    press_key("'")

#Select the Emulator
pyautogui.moveTo(20, 110, 0.5)
time.sleep(0.3)
pyautogui.moveTo(640, 110, 0.5)
time.sleep(0.3)
pyautogui.moveTo(640, 580, 0.5)
time.sleep(0.3)
pyautogui.moveTo(20, 580, 0.5)
time.sleep(0.3)
#pyautogui.click()
#Play the Game
#select_next_unit()
#move_unit(left=2, up=2)

