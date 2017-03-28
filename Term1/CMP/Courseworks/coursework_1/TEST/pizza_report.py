"""
Sari Nusier
1317015

Description: Solution to Coursework 1 Part 2

Input: data/pizza.xml file which describes the pizza menu of a specific
restaurant.

Output: data/pizza_report.csv file containing the number of available pizza
sizes, toppings, crusts and the total number of possible pizzas that can be
created.
"""

import xml.etree.ElementTree as et
import csv


try:
    # reading file and setting root of xml tree
    pizza_menu_tree = et.ElementTree(file='data/pizza.xml')
    pizza_menu_root = pizza_menu_tree.getroot()

    # totals are stored into a dictionary
    totals = {}

    # number of sizes available
    totals["sizes"] = len(pizza_menu_root.find("sizes"))

    # number of toppings available
    totals["toppings"] = len(pizza_menu_root.find("toppings"))

    # number of crusts available
    totals["crusts"] = len(pizza_menu_root.find("crusts"))

    # number of possible pizzas
    totals["total_combo"] = totals["sizes"]*(2**totals["toppings"])*totals["crusts"]

    # writing the results into the csv file
    with open("data/pizza_report.csv", "w") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=['sizes','toppings','crusts','total_combo'])
        writer.writeheader()
        writer.writerow(totals)
except IOError as ioe:
    print("I/O Error: %s" % ioe)
