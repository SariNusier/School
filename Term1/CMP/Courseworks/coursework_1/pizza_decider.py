import xml.etree.ElementTree as et
import csv
import json

class PizzaRequest():
    def __init__(self,username,text,received_pizza=None):
        self.username = username
        self.text = text
        self.received_pizza = received_pizza

class PizzaRequestHistory():
    def __init__(self,requests):
        self.requests = requests

    def getRequestsByUsername(self,username):
        return [request for request in self.requests if username == request.username]

class PizzaDecider():
    def __init__(self,history):
        self.history = history

    def decide(self,request):
        if(len(request.text) <= 400):
            return False

        return not(any([r.received_pizza for r in self.history.getRequestsByUsername(request.username)]))



try:
    with open("data/random_acts_pizza.csv", "r") as csvFile:
        csvRequests = csv.DictReader(csvFile)
        request_history = PizzaRequestHistory([PizzaRequest(csvReq["username"],
                                                           csvReq["text"],
                                                           csvReq["received_pizza"])
                                              for csvReq in csvRequests])

    with open("data/pizza_request.json","r") as requestFile:
        request_from_file = PizzaRequest(**json.load(requestFile))

    decider = PizzaDecider(request_history)
    request_from_file.received_pizza = decider.decide(request_from_file)

    with open("data/pizza_decision.json","w") as decisionFile:
        json.dump(request_from_file.__dict__,decisionFile)



except IOError as ioe:
    print("I/O Error: %s" % ioe)
