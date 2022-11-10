# **HAutoML-Bench** 

**HAutoML-Bench**  es un benchmark de AutoML heterogéneo. Es un sistema extensible , de código abierto y que aún continúa en desarrollo. Permite la descarga de datasets y la inclusión de otros. Además contiene funcionalidades para cuantificar el rendimiento en los mismos y para filtrarlos según sus metadatos.

En el directorio source archivo test se provee algunos ejemplos de uso  para su fácil utilización.

###  Instrucciones de Instalación e Inicialización

1. Para su instalación primero descargarlo desde su repositorio oficial de github. 

2.  Instalar poetry un gestor de paquetes mediante el siguiente comando:  

   ``` 
   pip3 install poetry
   ```

3. Instalar todas las dependencias del benchmark ejecutando el siguiente comando en la carpeta root del proyecto : 

   ```
   poetry install
   ```

4. Luego debe colocar su archivo de pruebas en la carpeta source e importar la clase HAutoMLBench

   ```
   from benchmark import HAutoMLBench
   ```

5. Llamar al método create_datsets de la siguiente forma el cual creará el estado inicial del benchmark:

    ```
   HAutoMLBench.create_datsets()
    ```

### Directorio 

En la figura \ref{fig:image1} se muestra la estructura de sus archivos inicialmente. Luego de ejecutar el metodo init se crean dos nuevos archivos que guardan el estado del benchmark, infos y names . Tambien se crea una carpeta dataset que contiene todos los dataset serializados y sus funciones load.

### Clase  HAutoMLBench y Dataset

El benchmark intervienen dos clases : HAutoMLBench y Dataset. 

La clase HAutoMLBench es la encargada de interactuar con el usuario , en ella se encuentran todos los métodos que permiten probar los sistemas. Estos son estáticos para no necesitar instanciar la clase para utilizar sus funcionalidades. En la subsección tal se discuten cada uno de ellos. 

La clase Dataset por otro lado es la encargada de modelar cada uno de los datasets del benchmark y almacenar sus propiedades. Entre estas últimas está el nombre,  url  utilizada para la descarga de su contenido , informaciones de la naturaleza de sus datos y una función cargadora o llamada function_loader. Esta función es la que permite acceder al contenido de cada dataset. Dicha clase hereda de la clase YamlAble . Esto le brinda la funcionalidad de serializarse como un objeto .yaml, permitiendo guardar cada uno de sus valores y acceder a ellos de forma rápida y  fácil. 

**Propiedades de un Dataset**

La propiedad info almacena algunos metadatos del dataset. El numero de columnas, el numero de instancias del conjunto de entrenamiento y de prueba. También la tarea que resuelve y el numero de valores null que posee. Se brinda para cada uno el tipo semántico inferido para cada una de sus columnas. Esta inferencia se realizó a partir de la  descripción en el dataset original. Por  ultimo también se tiene un metadato para el numero de clases en tareas de clasificación y el balance de las mismas medido por el numero de instancias de la clase minoritaria frente a la mayoritaria.    

```
class Dataset:
	url: str
	name: str
    function_loader : function
    info : dict {'n_instances': [int,int],
          		 'n_columns': int , 
          		 'columns_type': {'name' : semantic_type},
                 'targets': list[str] ,
                 'null_values': int,
                 'task': str 
                 'pos_label': Any,
                 'classes': int, 
                 'class balance':float}
  
```

**Tipos semánticos**: 

Para las entradas de texto :  

text: Posee más de una idea que pueden o no tener relación.  

sentence: Es un texto que transmite una idea esencial, la cual no puede ser separada en partes para su entendimiento.

word:  Tiene significado propio es la más pequeña entidad

syntagma : No es considerado una palabra pues puede ser la unión de mas de una de estas. Es una frase que no puede dividirse en palabras porque pierde su significado , pero a la vez no llega a ser una oración porque no indica acción ni estado. 

Para las entradas numéricas:

int: numero entero.

float: numero decimal.

Variables categóricas:

category: La variable puede ser dividida en tipos o categorías.

Variables booleanas:

boolean: Se tomaron solo como variables booleanas aquellas que tienen valores True o False.

Variables de tiempo :

datetime: Columnas que especifican una fecha o un tiempo.

Variables de localización:  

url: los enlaces a otros datos.

path_image: el enlace a una imagen o el path de una carpeta de imágenes.

Variables relacionadas a reconocimiento de entidades:

SeqTokens : una lista de palabras a etiquetar como entidad .

SeqLabels: una lista de etiquetas que corresponden al formato IOB2. 

**Function_loader **

Todos los datasets poseen una función loader que le permite descargar , abrir  y leer el contenido propio de cada uno de ellos.

Cada una de estas funciones toman la mismas  entradas y depende de ellas la salida. El contenido puede ser devuelto en formato Dataframe de pandas o una lista que contiene en forma de listas las instancias, se especifica con el parámetro format. Puede escoger si utilizar el dataset entero o dividido en entrenamiento y prueba , mediante el parámetro samples. En caso de que desee separar los conjuntos tanto de entrenamiento como de prueba en la X y la y, utilice el parámetro booleano _in_x_y.  Al leer los archivos puede introducir la codificación 'utf-8' es la por defecto, se recomienda que este parámetro no se cambie. Por ultimo el target o nombre de la etiqueta de salida del modelo se recomienda solo cambiar este parámetro por los targets especificados en la info del dataset .  El usuario no trabaja directamente con estas funciones , sino con una general ubicada en la clase HAutoMLBench  que le permite hacer load a cualquier datasets mediante su nombre y todos los parámetros antes mencionados

```
def function_loader(self,format = "pandas", in_x_y = True ,samples =  2, encoding = "utf-8",target = "label")
'''
input: self: Dataset
	   format: str
	   in_x_y : bool
	   samples : int
	   encoding: str
	   target: str
output: tuple	   
if in_x_y = True ,samples =  2  return X_train, y_train ,X_test, y_test
   in_x_y = True ,samples =  1 return X_all, y_all
   in_x_y = False samples =  2  return train, test
   format = "pandas", in_x_y = False, samples =  1  return all
   
   pandas : table 
   list : list [[intance1],[intance2]....]
'''
```



### Métodos de interacción del usuario

**Método create-datasets** 

El método create_datasets como se mencionó anteriormente en las instrucciones de instalación se encarga de guardar el estado inicial del benchmark . Para ello almacena en archivos los nombres, urls, las informaciones  y los nombres de las funciones load de todos los dataset originales . Además se encarga de crear las instancias de los mismos y serializarlas para su uso futuro. Es importante que solo se ejecute esta función una sola vez, debido a que se perderán todos los cambios guardados y volverá a su estado inicial por defecto. No posee parámetros de entrada y no retorna ningún valor . Si ocurre algún error lo imprime en pantalla.

**Método init**

El método init devuelve el estado actual del benchamrk , retorna 3 salidas por defecto, los nombres,urls y nombres de las funciones loader de todos los datasets incluidos hasta el momento. Posee un parámetro opcional booleano , que si es verdadero retorna una cuarta salida que son las informaciones o info.

```
def init(ret_info = False):
	'''
	Input: ret_info: bool 
	Output: tuple : 
    	names: [str]
    	urls: [str]
    	functions_name: [str]
    	infos: dict {key = name : value =info}
   
	'''
```

**Método new_dataset**

El método new-datset se encarga de agregar un nuevo dataset al benchmark, del mismo debe proveer su nombre, url de donde se encuentra y  su función loader. Como parámetro no obligatorio tiene  sus metadatos, que sería la llamada info. Esta info debe tener la misma estructura que la especificada en la clase dataset. Si desconoce algunas de las propiedades presentes en ella márquelas como None.  La función load debe estar en el archivo *functions_load.py* y si necesita descargar el dataset desde una url puede utilizar la función download de la clase Dataset. 

Si existe un dataset con el mismo nombre se remplazará y si ocurre algún error durante la ejecución producto a problemas con la introducción de la funcion loader o con la info se restablece a sus valores por defecto el benchmark. 

Todos los errores se imprimen en pantalla.

Parámetros por defecto : 

```
def new_dataset(name: str, url: str, function: str or function, info = None)
	'''
	Non-mandatory parameters: info: dict{}
	output : None
	''
```

**Método remove_dataset**

Se encarga de remover el dataset que coincida con el nombre que se pasa como parámetro de entrada.  Lanza error en caso de que ocurra algún problema durante la escritura de los archivos del estado del benchmark. 

  Parámetros por defecto : 

```
def remove_dataset(name):
    '''
    input: name: str
    output: None
    '''
```

**Método load_dataset**

Esta es la función a la cual se hizo referencia anteriormente, permite la descarga y utilización del contenido de los datasets.  El único parámetro obligatorio es el nombre,si los otros no se introducen se toma su configuración  por defecto. Su objetivo es  buscar el  datasets que tiene como nombre el introducido y luego llamar a la  function_loader del mismo . Debido a  esto la forma de la salida es la misma que la de  function_loader

```
def load_dataset(cls, name, format = "pandas", in_xy = True, samples = 2,encoding = 'utf-8',target = None):

```

**Método load_info **

Dado el nombre de un dataset carga su información y la retorna . Error si el dataset no se existe. La salida tiene la misma forma que la info descrita en la clase dataset , pero si ocurre un error se retorna None.

```
def load_info(cls,name):
'''
input: name:str
output : info : dict o None
'''
```

**Método filter**

El objetivo del método filter es devolver una lista de datasets que cumpla con las restricciones de los parámetros de entrada. La primera restricción que puede ser introducida es el tipo de tarea. El segundo parámetro es el cumplimiento de una propiedad numérica del dataset, esta propiedad debe estar presente en la información.  En caso de no establecer ninguna restricción se devuelve una lista con todos los datasets introducidos  

```
def filter(cls, task = None,expresion = None):
'''
input:
    task: str = 'binary', 'regression,'multiclass'
    expression: tuple(len(3)) = (property,min,max) : min <= property < max
    property: str = 'n_instances','n_columns', 'null_values','classes','class balance'

output: list[str]
'''  
```

**Método evaluate** 

El método evaluate es el encargado de medir el rendimiento de un sistema AutoML. Recibe como entrada los datos del target  del test y los valores predictivos obtenidos con el modelo sobre el conjunto test. También el tipo de tarea , el nombre del archivo y la ruta en donde se quieren guardar los resultados. Además recibe una lista de los distintos labels del target , debería siempre introducirlo cuando la tarea sea clasificación multiclase y el target sea de tipo no numérico . EL parámetro pos debe introducirse cuando en la clasificación binaria el target es no numérico y así especificar la clase positiva.  El método utiliza las métricas implementadas en el  paquete de python sklearn. Según el tipo de tarea se evalúan sus métricas correspondientes.

La salida es un diccionario que tiene como llave cada una de las métricas y como value el valor obtenido para cada una de ellas. Durante la ejecución se escribe un archivo json con los resultados obtenidos en el dataset. Este archivo es capaz de almacenar todos los resultados de un sistema en todos los datasets .

El archivo tiene la forma de: 

```
dict{key: name_datasets, value: {key: name_metric,value: value_metric}
```

```
def evaluate(cls,name,y_true,y_pred, task, pos= None ,labels = None,save_path = None, name_archive ='results'):
'''
input: name: str
	   y_true: array, list
	   y_pred: array, list
	   task: str
Non-mandatory parameters input:
	   pos: str
	   labels: list[str]
	   save_path: str : path
	   name_archive: str
output: 
	result: {key: name_metric,value: value_metric}
'''
```

 ### Métodos auxiliares 

Existen otros métodos que fueron de utilidad para implementar las anteriores funcionalidades descritas. Ejemplo :  __update_info__, write_info,  __change_name entre otros .No debe utilizar ninguno de estos métodos pues puede perturbar la integridad del benchmark.