#!/bin/bash                    
#SBATCH -J RNET28Dec       # Nombre del trabajo

#SBATCH --mail-type=ALL        # Correo de inicio, de error y de término
#SBATCH --mail-user=jorge.cardenas.l@mail.pucv.cl     # Correo del usuario

#SBATCH --account=research     # Asociación de la cuenta
#SBATCH --partition=full       # Nombre de la partición a usar

#SBATCH -n 1                   # Número total de nodos
#SBATCH -c 1                  # Número de núcleos por tarea

#SBATCH --gres=gpu:0        # Asignar el recurso de GPU a utilizar (en GB)
#SBATCH --mem-per-cpu=4250     # Memoria por CPU (en MB)

#SBATCH --output=output_29Ene_RESNET18.txt    # Archivo de salida estándar
#SBATCH --error=error_29ENE_RESNET18.txt      # Archivo de error estándar

##################################################################################
# Instrucciones para ejecutar el trabajo del usuario (Ejemplo Python)

# Activar ambiente de trabajo
source activate metasurfacesEnv                         # Activar Conda
cd metasurface_AI_V2/Predictor/RESNET
python main_TM_V5_customSet_highAbs_512.py                          # Mandar a Slurm script de Python
conda deactivate                         # Desactivar el entorno específico