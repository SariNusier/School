"""
Sari Nusier
1317015

Description: Solution to Coursework 1 Part 4

Input:
data/random_acts_pizza.xml: file which contains the
history of pizza requests

data/pizza_request.json: file containing the request to be analysed.


Output: data/pizza_decision.json containing the request's username, text,
and the decision made.
"""

import csv
import json

class PizzaRequest():
    """This object describes a pizza request containing the username of the
    requestor, the text of the request and a boolean to tell if the request was
    accepted or not"""
    def __init__(self,username,text,received_pizza=None):
        self.username = username
        self.text = text
        self.received_pizza = received_pizza

class PizzaRequestHistory():
    """Object that helps define a history of requests (using a list)"""
    def __init__(self,requests):
        self.requests = requests

    def getRequestsByUsername(self,username):
        """This method returns all the requests made by a specific user"""
        return [request for request in self.requests
                if username == request.username]

class PizzaDecider():
    """This class is used to decide whether a user's request should be accepted
    or not, based on the rules defined in the coursework."""
    def __init__(self,history):
        self.history = history

    def decide(self,request):
        """Method returns a boolean decision on whether the request is
        acceptable or not"""
        #length rule: only texts over 400 characters should be accepted
        if(len(request.text) <= 400):
            return False

        """if no accepted request (PizzaRequest.received_pizza == True) can be
        found in the history of a specif user's requests, then the return True,
        otherwise return false.
        any() will return True if it finds a True in the array of requests from
        the user, hence the need for a not() (return True if none is found)
        """
        return not(any([r.received_pizza
                for r in self.history.getRequestsByUsername(request.username)]))



try:
    # reading the history of requests
    with open("data/random_acts_pizza.csv", "r") as csvFile:
        csvRequests = csv.DictReader(csvFile)
        request_history = PizzaRequestHistory([PizzaRequest(csvReq["username"],
                                                           csvReq["text"],
                                                           csvReq["received_pizza"])
                                              for csvReq in csvRequests])


    # instantiate the decider with the history read from file
    decider = PizzaDecider(request_history)

    # reading the request to be tested
    with open("data/pizza_request.json","r") as requestFile:
        request_from_file = PizzaRequest(**json.load(requestFile))

    # the current received_pizza state is changed to the decision.
    request_from_file.received_pizza = decider.decide(request_from_file)

    # write the updated object to a json file
    with open("data/pizza_decision.json","w") as decisionFile:
        json.dump(request_from_file.__dict__,decisionFile)


except IOError as ioe:
    print("I/O Error: %s" % ioe)
