# <div align="center">FULBO</div>

### <div align="center">Desarrollo de sistema de captura y procesamiento de video con técnicas de machine learning aplicadas a partidos de fútbol</div>

<div align="center">Agustín Roldán y Germán Katzenelson</div>

---

## Cómo Configurar el Proyecto

Este proyecto utiliza un entorno virtual (`venv`) para manejar las dependencias de Python y un archivo `requirements.txt` para compartir las mismas entre los desarrolladores. A continuación, te explicamos cómo configurarlo correctamente.

---

### **1. Crear el entorno virtual**

Después de clonar el repositorio, crea un entorno virtual en tu máquina. Asegúrate de estar en el directorio raíz del proyecto antes de ejecutar este comando:

```bash
python -m venv venv
```

---

### **2. Activar el entorno virtual**

Activa el entorno virtual según tu sistema operativo:

- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

---

### **3. Instalar las dependencias**

Si su placa de video es compatible puede instalar CUDA para agilizar el procesamiento:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Luego, ya con el entorno virtual activo, instala las dependencias especificadas en el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### **4. Agregar nuevas dependencias**

Si necesitas instalar una nueva librería, hacelo con `pip install`. Por ejemplo:

```bash
pip install nombre_libreria
```

Luego, actualiza el archivo `requirements.txt` para que otros puedan instalar las mismas dependencias:

```bash
pip freeze > requirements.txt
```

Subí el archivo actualizado al repositorio.

---

### **5. No subir el entorno virtual al repositorio**

Para evitar que el entorno virtual (`venv/`) se suba al repositorio, asegurate de que el archivo `.gitignore` contenga la siguiente línea (debería estar):

```plaintext
venv/
```

Esto evita que los archivos innecesarios ocupen espacio en el repositorio.
