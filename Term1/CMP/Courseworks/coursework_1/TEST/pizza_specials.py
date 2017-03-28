"""
Sari Nusier
1317015

Description: Solution to Coursework 1 Part 3

Input:
data/pizza.xml: file which describes the pizza menu of a specific restaurant.
data/pizza_specials.csv: file containing the special pizzas' descriptions.


Output: data/pizza_specials.txt containing the special pizzas' descriptions in a
more human readable format.
"""

import xml.etree.ElementTree as et
import csv


def formatStringToPrint(name, size, toppings, crust):
    """helper function that builds the string to be written into
    pizza_specials.txt"""
    # string that will contain all the toppings separated by an and
    allToppings = ""

    # adding toppings to allToppings
    for topping in toppings:
        allToppings += topping + " and "

    #returning the formatted string
    return "%s: %s Pizza with %s%s"%(name, size, allToppings, crust)


# this is a helper function that translates the toppings code into the corresponding names
def getToppings(toppingCodes, toppingString):
    """this is a helper function that translates the toppings code into the
    corresponding names"""
    return [toppingCodes[toppingString[c]] for c in range(0, len(toppingString))]


try:
    # reading file and setting root of xml tree
    pizza_menu_tree = et.ElementTree(file='data/pizza.xml')
    pizza_menu_root = pizza_menu_tree.getroot()

    # declaring the dictionaries to store the code and text pairs
    sizeCodes = {}
    toppingCodes = {}
    crustCodes = {}

    # getting sizes
    for size in pizza_menu_root.find("sizes"):
        sizeCodes[size.attrib["code"]] = size.text

    # getting toppings
    for topping in pizza_menu_root.find("toppings"):
        toppingCodes[topping.attrib["code"]] = topping.text

    # getting crusts
    for crust in pizza_menu_root.find("crusts"):
        crustCodes[crust.attrib["code"]] = crust.text

    # reading the specials
    with open("data/pizza_specials.csv", "r") as csvFile:
        csvSpecials = csv.DictReader(csvFile)

        # writing the specials to file.
        # PLEASE SEE formatStringToPrint()
        with open("data/pizza_specials.txt", "w") as outputFile:
            for special in csvSpecials:
                outputFile.write(formatStringToPrint(special["name"],
                                   sizeCodes[special["size"]],
                                   getToppings(toppingCodes, special["toppings"]),
                                   crustCodes[special["crust"]])+"\n")

except IOError as ioe:
    print("I/O Error: %s" % ioe)
