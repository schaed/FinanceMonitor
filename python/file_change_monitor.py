#!/usr/bin/python
import time,os
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Creates trades when requested
class  MyFileHandler(FileSystemEventHandler):
    
    def __init__(self,direct = '', loadFiles=True, debug=False):
        FileSystemEventHandler.__init__(self)
        self.directory = direct
        self.loadFiles = loadFiles
        self.debug = debug

        # This is a list of files that we want to do something if changed
        self.file_list=['/out_momentum_instructions.csv',
                   #'/out_target_instructions.csv',
                   #'/out_merger_instructions.csv',
                   #'/out_upgrade_instructions.csv',
                   #'/out_bull_instructions.csv',
                   #'/out_pharmaphase_instructions.csv'
        ]
        
        # initialize loading files
        if self.loadFiles:
            self.on_modified(True)
        
    def set_monitor_directory(self,dir_name):
        self.directory = dir_name.rstrip('/')+'/'
        
    def set_file_list(self, new_file_list):
        self.file_list = new_file_list
        
    def on_moved(self, event):
        print(f'event type: {event.event_type} path : {event.src_path}')
    def  on_modified(self,  event):
        #print(f'event type: {event.event_type} path : {event.src_path}')

        # if boolean, then load them all. otherwise, see if the file was modified
        if type(event)==type(True):
            for ifile_name in self.file_list:
                self._read_csv(in_file_name=self.directory+ifile_name)
        else:
            for ifile_name in self.file_list:
                if ifile_name in event.src_path:
                    self._read_csv(in_file_name=event.src_path)
                    
    def  on_created(self,  event):
        print(f'event type: {event.event_type} path : {event.src_path}')
    def  on_deleted(self,  event):
        print(f'event type: {event.event_type} path : {event.src_path}')

    def _read_csv(self, in_file_name='Instructions/out_target_instructions.csv'):
        """read_csv - reads in the csv files and applies basic sanity checks like that the price is not already above the new recommendation
        Inputs:
        in_file_name - str - input csv file path like Instructions/out_bull_instructions_test.csv
        """
        if not os.path.exists(in_file_name):
            print(f'File path does not exist! {in_file_name}. Skipping...')
            return
        
        # reading in the sets of trades that we should be executing
        dfnow=[]
        try:
            dfnow = pd.read_csv(in_file_name, sep=' ')
        except (ValueError,FileNotFoundError,ConnectionResetError,FileExistsError):
            print(f'Could not load input csv: {in_file_name}')
            dfnow=[]

        if self.debug:
            print(dfnow)
        return dfnow

def main(args):
    
    # checking for trades to execute!
    event_handler = MyFileHandler(args.path)
    observer = Observer(timeout=1)
    print(args.path)
    observer.schedule(event_handler,  path=args.path,  recursive=True)
    observer.start()

    run = 1
    while run:
        input('waiting...')
        run=0
    observer.stop()
    observer.join()
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('symbols', nargs='+')
    parser.add_argument('--lot', type=float, default=-1,help='The amount of cash to spend')
    parser.add_argument('--path', type=str, default="/home/schae/testarea/FinanceMonitor/Instructions/",help='The path to the directory to check')

    main(parser.parse_args())
