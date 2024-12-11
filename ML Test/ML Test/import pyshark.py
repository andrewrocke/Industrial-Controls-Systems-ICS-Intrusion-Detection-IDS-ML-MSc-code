import pyshark
import os
import csv
from scapy.utils import RawPcapReader


import pandas as pd
capcount = 0
file_in = "C:\\Users\\andre\\OneDrive\\Documents\\Dissertation\\research\\ICS Incident Response\\ML\\code\\ML Test\\ML Test\\veth665f3cf-0.pcap"
file_out = "out.csv"

cap = pyshark.FileCapture(file_in)
#cap.load_packets()
cap.reset()

#for packet in cap:
x=0
for packet in cap: 
    x=x+1
    print(x)
    #if x == 4:
     #break
    #print(cap.next_packet) 
    #from https://github.com/johnbumgarner/pyshark_usage_overview/blob/master/docs/parsing/ip_packets.md
    try:
        # obtain all the field names within the IP packets
      if x == 1:
        field_names = packet.ip._all_fields
        with open(file_out, 'w', newline='') as csvfile:
         csvwriter = csv.writer(csvfile)
         csvwriter.writerow(field_names)
    
    
        # obtain all the field values
         field_values = packet.ip._all_fields.values()
    #field_dict = 
        # enumerate the field names and field values
          #for field_name, field_value in zip(field_names, field_values):
            #data = f'{field_name}'
            #data1 = f'{field_value}'        
            #print(data)
            #print('\n')  
            #print(data1)
        with open(file_out, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(field_values)
    except: 
        print("error")      
