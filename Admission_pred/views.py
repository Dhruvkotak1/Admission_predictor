from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import joblib,math

# Load the model 
model = joblib.load("admission_prediction_model")


def home(request):
    return render(request, 'index.html')

# Get the data from website and predict chances of admission 
def getData(request):
    if(request.method=="GET"):
        try:
            # Converting the data into essential data types and storing them into variables
            gre = float(request.GET.get('gre'))
            toefl = float(request.GET.get('toefl'))
            university_rating = float(request.GET.get('university_rating'))
            sop = float(request.GET.get('sop'))
            lor = float(request.GET.get('lor'))
            cgpa = float(request.GET.get('cgpa'))
            research = float(request.GET.get('research'))
            print(gre, toefl, university_rating, sop, lor, cgpa, research)

            # Predicting the chances of admission
            chances=model.predict([[gre, toefl, university_rating, sop, lor, cgpa, research]])
            result=chances[0]*100
            print(result)

            # Return the result to the html page
            return JsonResponse({'status':1,'msg':result})
        except Exception as e:
            print(e)
            return JsonResponse({'status':0,'msg':'Invalid Input'})