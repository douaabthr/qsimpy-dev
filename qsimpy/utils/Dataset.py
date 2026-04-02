import ast
import pandas as pd


class Dataset:
    def __init__(self, circuit_file: str, errors_file: str):
        self.circuit_file = circuit_file
        self.errors_file = errors_file
        self.data = {}         
        self.errors_data = {}  
        self.load_data_pd(self.circuit_file)

        if self.errors_file:
                self.load_errors_data()
    # Load data from csv file using pandas
    def load_data_pd(self, filename):   
        self.df = pd.read_csv(filename)

        # Ensure 'subset_id' is the first column
        if "subset" in self.df.columns:
            cols = ["subset"] + [col for col in self.df.columns if col != "subset"]
            self.df = self.df[cols]

        # for index, row in self.df.iterrows():
        #     # Convert string representation of a dictionary to an actual dictionary
        #     for col in self.df.columns:
        #         if isinstance(row[col], str) and "{" in row[col] and "}" in row[col]:
        #             row[col] = ast.literal_eval(row[col])

        #     # Form nested dictionary structure
        #     formatted_data = {
        #         "algorithm": row["algorithm"],
        #         "original": {
        #             "width": int(row["original_width"]),
        #             "depth": int(row["original_depth"]),
        #             "gates": row["original_gates"],
        #         },
        #         "ibmq7": {
        #             "width": int(row["ibmq7_width"]),
        #             "depth": int(row["ibmq7_depth"]),
        #             "gates": row["ibmq7_gates"],
        #         },
        #         "ibmq16": {
        #             "width": int(row["ibmq16_width"]),
        #             "depth": int(row["ibmq16_depth"]),
        #             "gates": row["ibmq16_gates"],
        #         },
        #         "ibmq27": {
        #             "width": int(row["ibmq27_width"]),
        #             "depth": int(row["ibmq27_depth"]),
        #             "gates": row["ibmq27_gates"],
        #         },
        #         "ibmq127": {
        #             "width": int(row["ibmq127_width"]),
        #             "depth": int(row["ibmq127_depth"]),
        #             "gates": row["ibmq127_gates"],
        #         },
        #         # "arrival_time": row["arrival_time"],
        #     }

        #     # Index by algorithm and original_width
        #     key = (row["subset"], row["algorithm"], int(row["original_width"]))
        #     self.data[key] = formatted_data
            
            # Detect backends dynamically (washington, hanoi, etc.)
        all_prefixes = set(col.split('_width')[0] for col in self.df.columns if '_width' in col)
        backends = [p for p in all_prefixes if p != 'original']

        for _, row in self.df.iterrows():
            # Build the nested structure
            formatted_data = {
                "id_task": row["task_id"],
                "algorithm": row["algorithm"],
                "original": {
                    "width": int(row["original_width"]),
                    "depth": int(row["original_depth"]),
                    "gates": row["original_gates"],
                }
            }

            # Map all dynamic backends
            for b in backends:
                formatted_data[b] = {
                    "width": row.get(f"{b}_width"),
                    "depth": row.get(f"{b}_depth"),
                    "gates": row.get(f"{b}_gates_with_errors"),
                    "critical_path":row.get(f"{b}_critical_path"),
                    "readout_errors": row.get(f"{b}_readout_errors")
                }

            # This key MUST match what the Gym Env expects (subset, algorithm, width)
            subset_val = row.get("subset", 1) 
            key = (subset_val, row["algorithm"], int(row["original_width"]))
            self.data[key] = formatted_data



    # Load data from csv file using pandas
    def load_test_data_pd(self, filename):
        self.df = pd.read_csv(filename)

        # Ensure 'subset_id' is the first column
        if "subset" in self.df.columns:
            cols = ["subset"] + [col for col in self.df.columns if col != "subset"]
            self.df = self.df[cols]

        for index, row in self.df.iterrows():
            # Convert string representation of a dictionary to an actual dictionary
            for col in self.df.columns:
                if isinstance(row[col], str) and "{" in row[col] and "}" in row[col]:
                    row[col] = ast.literal_eval(row[col])

            # Form nested dictionary structure
            formatted_data = {
                "algorithm": row["algorithm"],
                "original": {
                    "width": int(row["original_width"]),
                    "depth": int(row["original_depth"]),
                    "gates": row["original_gates"],
                },
                "ibmq7": {
                    "width": int(row["ibmq7_width"]),
                    "depth": int(row["ibmq7_depth"]),
                    "gates": row["ibmq7_gates"],
                },
                "ibmq16": {
                    "width": int(row["ibmq16_width"]),
                    "depth": int(row["ibmq16_depth"]),
                    "gates": row["ibmq16_gates"],
                },
                "ibmq27": {
                    "width": int(row["ibmq27_width"]),
                    "depth": int(row["ibmq27_depth"]),
                    "gates": row["ibmq27_gates"],
                },
                "ibmq127": {
                    "width": int(row["ibmq127_width"]),
                    "depth": int(row["ibmq127_depth"]),
                    "gates": row["ibmq127_gates"],
                },
                "arrival_time": row["arrival_time"],
            }

            # Index by algorithm and original_width
            key = (row["subset"], row["algorithm"], int(row["original_width"]))
            self.data[key] = formatted_data


    def load_errors_data(self):
        if not self.errors_file:
            return

        df = pd.read_csv(self.errors_file)

        # Convertir toutes les colonnes dict
        for col in ["gates_errors", "readout_errors", "critical_path"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: ast.literal_eval(x) if pd.notna(x) else {}
                )

        # Indexer
        self.errors_data = {
            (str(row["task_id"]), row["backend"]): {
                "gate_errors": row.get("gates_errors", {}),
                "readout_errors": row.get("readout_errors", {}),
                "critical_path": row.get("critical_path", {})
            }
            for _, row in df.iterrows()
        }
    def get_subset_data(self, subset_id):

        return {key: value for key, value in self.data.items() if key[0] == subset_id}
    
    def get_test_subset_data(self, subset_id):
        # Filter data by subset_id
        self.load_test_data_pd(self.circuit_file)
        return {key: value for key, value in self.data.items() if key[0] == subset_id}
    
    def get_errors(self, task_id, backend):
        return self.errors_data.get((str(task_id), backend), {"gate_errors": {}, "readout_errors": {}})

    def get_gate_times(self, task_id, backend):
        data = self.errors_data.get(
            (str(task_id), backend),
            {}
        )
        return data.get("critical_path", {})