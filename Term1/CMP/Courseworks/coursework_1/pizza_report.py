import xml.etree.ElementTree as et
import csv


try:
    # reading file and setting root of xml tree
    tree = et.ElementTree(file='data/pizza.xml')
    root = tree.getroot()

    # totals are stored into a dictionary
    result = {}

    # number of sizes available
    result["sizes"] = len(root.find("sizes"))

    # number of toppings available
    result["toppings"] = len(root.find("toppings"))

    # number of crusts available
    result["crusts"] = len(root.find("crusts"))

    # number of possible pizzas
    result["total_combo"] = result["sizes"]*(2**result["toppings"])*result["crusts"]

    # writing the results into the csv file
    with open("data/pizza_report.csv", "w") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=['sizes','toppings','crusts','total_combo'])
        writer.writeheader()
        writer.writerow(result)
except IOError as ioe:
    print("I/O Error: %s" % ioe)
