import xml.etree.ElementTree as et


try:
    # reading file and setting root of xml tree
    tree = et.ElementTree(file='data/pizza.xml')
    root = tree.getroot()

    # printing shop name
    print("# %s\n"%root.find("shopname").text)

    # printing sizes
    print("## Sizes")
    for size in root.find("sizes"):
        print("- %s"%size.text)

    # printing toppings
    print("\n## Toppings")
    for topping in root.find("toppings"):
        print("- %s"%topping.text)

    # printing crusts
    print("\n## Crusts")
    for crust in root.find("crusts"):
        print("- %s"%crust.text)
        
except IOError as ioe:
    print("I/O Error: %s" % ioe)
