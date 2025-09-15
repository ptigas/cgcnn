#!/usr/bin/env python3
"""
Example script showing how to use CGCNN to predict properties from ASE Atoms objects.

This script demonstrates:
1. Creating simple crystal structures using ASE
2. Using the CGCNNPredictor to make predictions
3. Batch predictions on multiple structures

Requirements:
- A trained CGCNN model (checkpoint file)
- atom_init.json file (element embeddings)
"""

import os
import numpy as np
from ase import Atoms
from ase.build import bulk
from cgcnn.predictor import CGCNNPredictor, predict_from_atoms


def create_example_structures():
    """
    Create some example crystal structures using ASE.
    
    Returns
    -------
    list of ase.Atoms
        List of example crystal structures
    """
    structures = []
    
    # Create simple cubic structures
    structures.append(bulk('Si', 'diamond', a=5.43))  # Silicon
    structures.append(bulk('Al', 'fcc', a=4.05))      # Aluminum
    structures.append(bulk('Fe', 'bcc', a=2.87))      # Iron
    
    # Create a simple binary compound (NaCl structure)
    nacl = bulk('NaCl', 'rocksalt', a=5.64)
    structures.append(nacl)
    
    # Create a supercell of silicon
    si_supercell = bulk('Si', 'diamond', a=5.43).repeat((2, 2, 2))
    structures.append(si_supercell)
    
    return structures


def main():
    """
    Main prediction example.
    """
    # Check if pre-trained model exists
    model_dir = 'pre-trained'
    model_files = []
    
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.pth.tar'):
                model_files.append(os.path.join(model_dir, file))
    
    if not model_files:
        print("No pre-trained models found in 'pre-trained' directory.")
        print("Please download a pre-trained model or train your own model first.")
        print("\nTo use this script, you need:")
        print("1. A trained CGCNN model (.pth.tar file)")
        print("2. An atom_init.json file with element embeddings")
        print("\nExample usage once you have a model:")
        print("python example_prediction.py --model path/to/model.pth.tar")
        return
    
    # Use the first available model
    model_path = model_files[0]
    print(f"Using model: {model_path}")
    
    # Look for atom_init.json in the model directory
    atom_init_file = os.path.join(os.path.dirname(model_path), 'atom_init.json')
    
    # Alternative locations to look for atom_init.json
    if not os.path.exists(atom_init_file):
        potential_paths = [
            'data/sample-regression/atom_init.json',
            'data/sample-classification/atom_init.json',
            'atom_init.json'
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                atom_init_file = path
                break
        else:
            print("atom_init.json file not found. Please ensure it's available.")
            print("Checked locations:")
            for path in [atom_init_file] + potential_paths:
                print(f"  - {path}")
            return
    
    print(f"Using atom initialization file: {atom_init_file}")
    
    try:
        # Initialize the predictor
        print("Initializing CGCNN predictor...")
        predictor = CGCNNPredictor(
            model_path=model_path,
            atom_init_file=atom_init_file,
            max_num_nbr=12,
            radius=8,
            dmin=0,
            step=0.2
        )
        
        print(f"Successfully loaded CGCNN model!")
        print(f"Model type: {'Classification' if predictor.model.classification else 'Regression'}")
        print(f"Model architecture:")
        print(f"  - Atom feature length: {predictor.model.embedding.out_features}")
        print(f"  - Number of conv layers: {len(predictor.model.convs)}")
        print(f"  - Hidden feature length: {predictor.model.conv_to_fc.out_features}")
        print(f"  - Number of hidden layers: {len(getattr(predictor.model, 'fcs', []))}")
        print(f"  - Output dimension: {predictor.model.fc_out.out_features}")
        
        # Create example structures
        print("\nCreating example crystal structures...")
        structures = create_example_structures()
        structure_names = ['Si (diamond)', 'Al (fcc)', 'Fe (bcc)', 'NaCl (rocksalt)', 'Si supercell (2x2x2)']
        
        # Make predictions
        print("\nMaking predictions...")
        print("-" * 60)
        
        for i, (atoms, name) in enumerate(zip(structures, structure_names)):
            try:
                prediction = predictor.predict(atoms)
                print(f"{name:20s}: {prediction:.4f}")
            except Exception as e:
                print(f"{name:20s}: Error - {str(e)}")
        
        # Batch prediction example
        print(f"\nBatch prediction for all {len(structures)} structures:")
        try:
            batch_predictions = predictor.predict_batch(structures)
            print(f"Batch predictions: {batch_predictions}")
        except Exception as e:
            print(f"Batch prediction failed: {str(e)}")
        
        # Example using convenience function
        print(f"\nUsing convenience function for single prediction:")
        try:
            single_pred = predict_from_atoms(structures[0], model_path, atom_init_file)
            print(f"Si (diamond) prediction: {single_pred:.4f}")
        except Exception as e:
            print(f"Convenience function failed: {str(e)}")
            
    except Exception as e:
        print(f"Failed to initialize predictor: {str(e)}")
        print("Please check that your model file and atom_init.json are compatible.")


def demo_custom_structure():
    """
    Demonstrate prediction on a custom structure.
    """
    print("\n" + "="*60)
    print("CUSTOM STRUCTURE EXAMPLE")
    print("="*60)
    
    # Create a custom perovskite-like structure
    # This is just an example - in practice you'd load real structures
    positions = np.array([
        [0.0, 0.0, 0.0],  # A-site
        [0.5, 0.5, 0.5],  # B-site  
        [0.5, 0.5, 0.0],  # O
        [0.5, 0.0, 0.5],  # O
        [0.0, 0.5, 0.5],  # O
    ])
    
    cell = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0], 
        [0.0, 0.0, 4.0]
    ])
    
    # Create atoms object (using Ca, Ti, O as example)
    atoms = Atoms(
        symbols=['Ca', 'Ti', 'O', 'O', 'O'],
        positions=positions,
        cell=cell,
        pbc=True
    )
    
    print(f"Custom structure: {atoms.get_chemical_formula()}")
    print(f"Cell parameters: {atoms.cell.lengths()}")
    print(f"Number of atoms: {len(atoms)}")
    
    # Note: This would need a compatible model and atom_init.json
    # that includes Ca, Ti, and O embeddings
    print("\nTo predict on this structure, ensure your model was trained")
    print("on data containing Ca, Ti, and O elements.")


if __name__ == "__main__":
    print("CGCNN Prediction Example")
    print("=" * 40)
    
    main()
    demo_custom_structure()
    
    print("\n" + "="*60)
    print("TIPS:")
    print("- Ensure your model and atom_init.json are compatible")
    print("- The atom_init.json must contain embeddings for all elements in your structures")
    print("- For best results, use structures similar to your training data")
    print("- Check the model's task type (regression vs classification)")
    print("="*60)