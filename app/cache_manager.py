import pandas as pd
import os
import ast

class SolutionCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self._init_cache()

    def _init_cache(self):
        if not os.path.exists(self.cache_file):
            df = pd.DataFrame(columns=["rho", "p", "hospitals_indices", "fitness", "z_max", "unserved_demand", "Horizon", "LOS", "occ"])
            df.to_csv(self.cache_file, index=False)

    def get_solution(self, rho, p, horizon, los, occ):
        df = pd.read_csv(self.cache_file)
        # Ensure types
        df["rho"] = df["rho"].astype(float)
        df["p"] = df["p"].astype(int)
        df["Horizon"] = df["Horizon"].astype(int)
        df["LOS"] = df["LOS"].astype(int)
        df["occ"] = df["occ"].astype(float)
        
        match = df[
            (df["rho"] == float(rho)) & 
            (df["p"] == int(p)) &
            (df["Horizon"] == int(horizon)) &
            (df["LOS"] == int(los)) &
            (df["occ"] == float(occ))
        ]
        
        if not match.empty:
            row = match.iloc[0]
            return {
                "hospitals_indices": ast.literal_eval(row["hospitals_indices"]),
                "fitness": row["fitness"],
                "z_max": row["z_max"],
                "unserved_demand": row["unserved_demand"]
            }
        return None

    def save_solution(self, rho, p, hospitals_indices, fitness, z_max, unserved_demand, horizon, los, occ):
        df = pd.read_csv(self.cache_file)
        # Ensure types for comparison
        df["rho"] = df["rho"].astype(float)
        df["p"] = df["p"].astype(int)
        df["Horizon"] = df["Horizon"].astype(int)
        df["LOS"] = df["LOS"].astype(int)
        df["occ"] = df["occ"].astype(float)
        
        # Check if exists
        mask = (
            (df["rho"] == float(rho)) & 
            (df["p"] == int(p)) &
            (df["Horizon"] == int(horizon)) &
            (df["LOS"] == int(los)) &
            (df["occ"] == float(occ))
        )
        
        if mask.any():
            # Update existing
            df.loc[mask, "hospitals_indices"] = str(list(hospitals_indices))
            df.loc[mask, "fitness"] = float(fitness)
            df.loc[mask, "z_max"] = float(z_max)
            df.loc[mask, "unserved_demand"] = float(unserved_demand)
        else:
            # Append
            new_row = {
                "rho": float(rho),
                "p": int(p),
                "hospitals_indices": str(list(hospitals_indices)),
                "fitness": float(fitness),
                "z_max": float(z_max),
                "unserved_demand": float(unserved_demand),
                "Horizon": int(horizon),
                "LOS": int(los),
                "occ": float(occ)
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
        df.to_csv(self.cache_file, index=False)
