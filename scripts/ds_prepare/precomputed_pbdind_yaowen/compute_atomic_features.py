from geometry_processors.process.sasa_calculator import SASASingleCalculator


calc = SASASingleCalculator("/scratch/sx801/temp/1a30_protein.pdb" ,"/vast/sx801/temp_sasa")
calc.run()