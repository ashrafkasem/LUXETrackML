from core.data.luxe_connector import LUXEDataConnector

if __name__ == "__main__":
    LUXEData = LUXEDataConnector(
        data_dir = "/home/amohamed/dust/amohamed/HTC/dataframes_new",
        output_dir = "output_luxe_data",
        start_event= 1,
        end_event= 5
        
    )
    dataFrame = LUXEData.readfiles()