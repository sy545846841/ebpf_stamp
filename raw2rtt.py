import argparse
import logging
import os
import re
import pandas as pd
from sktime.transformations.series.outlier_detection import HampelFilter


def parse_rtt_by_flow(data, sampling_rate):
    # calculate rtt
    data["rtt_s"] = (data["reply_rx"] - data["test_tx"])

    # Calculate time window
    min_timestamp = data["test_tx"].min()
    data["Time"] = ((data["test_tx"] - min_timestamp) // sampling_rate) * sampling_rate
    
    # Get session IDs
    ssids = data["ssid"].unique()

    # Calculate average per flow
    res_df = data[["Time", "rtt_s"]]
    res_df = res_df.groupby(res_df['Time']).mean().reset_index()

    for ssid in ssids:
        temp_df = data.loc[data['ssid'] == ssid][["Time", "rtt_s"]]
        res_df[ssid] = temp_df.groupby(temp_df['Time']).mean()
    
    res_df = res_df.rename(columns={"rtt_s": "Aggregate-Flow"})
    
    return res_df

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=str, help="Path to the directory of raw files")
    parser.add_argument("out_dir", type=str, help="Path to output directory")
    parser.add_argument("--sampling_rate", type=float, default=1.0, help="The sampling rate for the RTT value")
    parser.add_argument("--sep", type=str, default=',', help="Separator used for the output file" )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M')

    raw_files = os.listdir(args.src_dir)
    exist_files = os.listdir(args.out_dir)

    pattern = re.compile("\w+_raw.csv$")

    raw_paths = []
    rtt_paths = []
    raw_file_cont = 0
    existed_file_cont = 0

    for raw_f in raw_files:
        if len(pattern.findall(raw_f)) > 0:
            raw_file_cont += 1
            out_f = raw_f.replace("raw", "rtt")
            if out_f not in exist_files:
                raw_paths.append(os.path.join(args.src_dir, raw_f))
                rtt_paths.append(os.path.join(args.out_dir, out_f))
            else:
                existed_file_cont += 1
    
    logging.info("%d raw file(s) found" % raw_file_cont)
    logging.info("Parsing %d files (%d files arleady parsed)" % (len(raw_paths), existed_file_cont))
    
    for i in range(len(raw_paths)):
        # Parse RTT from raw data
        logging.info("[%d/%d] Collecting raw data form '%s'" % (i + 1, len(raw_paths), raw_paths[i]))
        
        raw_data_df = pd.read_csv(raw_paths[i])
        # Sort by test_pkt_tx_timestamp
        raw_data_df.sort_values(by=['test_tx'], inplace=True)
        raw_data_df.reset_index(drop=True, inplace=True)

        # Parse RTT
        rtt_data_df = parse_rtt_by_flow(raw_data_df, 1)

        # Apply Hampel filter to remove outliers
        col_names = list(rtt_data_df.keys())
        all_flows = col_names[:]
        all_flows.remove("Time")
        all_flows.remove("Aggregate-Flow")
        transformer = HampelFilter(window_length=10)
        for f in all_flows:
            rtt_data_df[f] = transformer.fit_transform(rtt_data_df[f])
        
        # Save to CSV
        rtt_data_df.to_csv(rtt_paths[i], index=False, sep=args.sep)
        logging.info("[%d/%d] RTT result saved to '%s'" % (i + 1, len(raw_paths), rtt_paths[i]))


if __name__ == "__main__":
    main()