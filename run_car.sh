#!/bin/bash

# Fonte o ambiente ROS 2
source ~/car_ws/install/setup.bash   # Fonte o ambiente ROS 2 do seu workspace
source ~/install/setup.bash           # Fonte o ambiente ROS 2 global (caso necessário)
source /opt/ros/foxy/setup.bash 
# Altere as permissões da porta serial para comunicação com o Arduino
sudo chmod 777 /dev/ttyACM0  # Ou a porta correta do seu dispositivo (verifique com 'ls /dev/tty*')

# Verifica se a porta serial está disponível antes de continuar
if [ ! -e /dev/ttyACM0 ]; then
    echo "Erro: A porta serial /dev/ttyACM0 não está disponível."
    exit 1
fi

# Inicia o nó de percepção (detecção de linha) em um novo terminal
gnome-terminal -- bash -c "ros2 run car_controller_pkg perception_node; exec bash"
echo "Perception iniciado."

# Inicia o nó de navegação
gnome-terminal -- bash -c "ros2 run car_controller_pkg navigation_node; exec bash"
echo "Navigation iniciado."

# Inicia o nó de controle (movimento do carro)
gnome-terminal -- bash -c "ros2 run car_controller_pkg controll_node; exec bash"
echo "Control iniciado."

# Exibe uma mensagem de sucesso
echo "Todos os nós foram iniciados com sucesso. Pressione Ctrl+C para parar os nós."

# Aguarda a interrupção do script (Ctrl+C)
wait

