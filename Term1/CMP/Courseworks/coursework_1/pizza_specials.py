import xml.etree.ElementTree as et
import csv


# helper function that builds the string to be written into pizza_specials.txt
def formatStringToPrint(name, size, toppings, crust):

    # string that will contain all the toppings separated by an and
    allToppings = ""

    # adding toppings to allToppings
    for topping in toppings:
        allToppings += topping + " and "

    #returning the formatted string
    return "%s: %s Pizza with %s%s"%(name, size, allToppings, crust)


# this is a helper function that translates the toppings code into the corresponding names
def getToppings(toppingCodes, toppingString):
    toppings = []

    # getting name character by character and appending to the array of names
    for c in range(0, len(toppingString)):
        toppings.append(toppingCodes[toppingString[c]])
    return toppings


try:
    # reading file and setting root of xml tree
    tree = et.ElementTree(file='data/pizza.xml')
    root = tree.getroot()

    # declaring the dictionaries to store the code and text pairs
    sizeCodes = {}
    toppingCodes = {}
    crustCodes = {}

    # getting sizes
    for size in root.find("sizes"):
        sizeCodes[size.attrib["code"]] = size.text

    # getting toppings
    for topping in root.find("toppings"):
        toppingCodes[topping.attrib["code"]] = topping.text

    # getting crusts
    for crust in root.find("crusts"):
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
