"""
Sari Nusier
1317015

Description: Solution to Coursework 1 Part 1

Input: data/pizza.xml file which describes the pizza menu of a specific
restaurant.

Output: the menu (to stdout) in the following format:

# (RestaurantName)

## Sizes
- (size1)
- (size2)
...

## Toppings
- (topping1)
- (topping2)
...

## Crusts
- (crust1)
- (crust2)
...

"""

import xml.etree.ElementTree as et


try:
    # reading file and setting root of xml tree
    pizza_menu_tree = et.ElementTree(file='data/pizza.xml')
    pizza_menu_root = pizza_menu_tree.getroot()

    # printing shop name
    print("# %s\n"%pizza_menu_root.find("shopname").text)

    # printing sizes
    print("## Sizes")
    for size in pizza_menu_root.find("sizes"):
        print("- %s"%size.text)

    # printing toppings
    print("\n## Toppings")
    for topping in pizza_menu_root.find("toppings"):
        print("- %s"%topping.text)

    # printing crusts
    print("\n## Crusts")
    for crust in pizza_menu_root.find("crusts"):
        print("- %s"%crust.text)

except IOError as ioe:
    print("I/O Error: %s" % ioe)
