

while True:
    command = input("Insert command: ")

    if command == "insert":
        command = input("What would you like to insert: ")

        if(command == "employee"):
            name = input("name: ")
            surname = input("surname: ")
            sex = input("sex: ")
            DOB = input("DOB: ")
            email = input("Email: ")
            print("INSERT INTO `employee` VALUES (NULL, '%s', '%s', '%s', '%s', '%s');" % (name, surname, sex, DOB, email))

        if(command == "status"):
            creatorID = int(input("creatorId: "))
            text = input("text: ")
            timestamp = input("timestamp: ")
            print("""INSERT INTO `status` VALUES (
              NULL,
              %d,
              '%s',
              '%s');""" % (creatorID, text, timestamp))

        if(command == "comment"):
            creatorId = int(input("creatorId: "))
            statusId = int(input("statusId "))
            text = input("text: ")
            timestamp = input("timestamp: ")
            print("""
            INSERT INTO `comment` VALUES (
              NULL,
              %d,
              %d
              '%s',
              '%s');
            """ % (creatorId, statusId, text, timestamp))

        
