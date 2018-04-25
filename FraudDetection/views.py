'''
Created on 21-Feb-2018

@author: Vishnu
'''

from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import permission_classes
from rest_framework import permissions
from .Classifier import model
from .CreditScoring import score
from .CreditScoringNew import newscore

@permission_classes((permissions.AllowAny,))
class Chance(viewsets.ViewSet):
    def create(self, request):
        question = request.data
        recommend = model(question['messageText'])
        result = {}
        if recommend == 1:
            result['chance'] = 'chance for default payment next month'
        else:
            result['chance'] = 'no chance for default payment next month'
        return Response(result)
    
@permission_classes((permissions.AllowAny,))
class Score(viewsets.ViewSet):
    def create(self, request):
        question = request.data
        Score = score(question['messageText'])
        result = {}
        result['score'] = Score
        return Response(result)
    
@permission_classes((permissions.AllowAny,))
class NewScore(viewsets.ViewSet):
    def create(self, request):
        question = request.data
        inputs = str(question['Seniority']) + ',' + str(question['Home']) + ',' + str(question['Time']) + ',' + str(question['Age']) + ',' + str(question['Marital']) + ',' + str(question['Records']) + ',' + str(question['Job']) + ',' + str(question['Expenses']) + ',' + str(question['Income']) + ',' + str(question['Assets']) + ',' + str(question['Debt']) + ',' + str(question['Amount']) + ',' + str(question['Price'])
        Score = newscore(inputs)
        result = {}
        result['score'] = Score
        return Response(result)
