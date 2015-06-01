#!/usr/bin/python

import pymetar
import paho.mqtt.publish as publish
import time 

#System Variables
station = 'KSSF'




def loadStation(station):
	rf=pymetar.ReportFetcher(station)
	rep=rf.FetchReport()
	return rep

def parseTemp(rep):
	rp=pymetar.ReportParser()
	pr=rp.ParseReport(rep)
	temp = pr.getTemperatureFahrenheit()
	return temp

def parseTime(rep):
	rp=pymetar.ReportParser()
	pr=rp.ParseReport(rep)
	time = pr.getISOTime()
	return time

def main():
	lasttime = []
	while 0 < 1:
		rep = loadStation(station)
		temp = parseTemp(rep)
		thime = parseTime(rep)
		if thime != lasttime:
			publish.single("/SanAntonioTemp", payload=("At "+str(thime)+" the temp was "+str(temp)),qos=2,hostname="10.0.0.3")
			lasttime = thime
			count = 0
			print "At "+str(thime)+ " the temp was "+str(temp)+ " has been published"  
		else:
			count = count + 15
			publish.single("/SanAntonioTemp", payload=("Time last updated was "+str(count/60)+ " minutes ago and the temperature was "+str(temp)),qos=2,hostname="10.0.0.3")
			print "Published new count " +str(count/60)
		
		time.sleep(15)


 

main()

