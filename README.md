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

5. Llamar al método *init* de la siguiente forma el cual creará el estado inicial del benchmark:

    ```
   HAutoMLBench.init()
   ```

### Clase  HAutoMLBench y Dataset

*HAutoML-Bench* está formado por dos clases principales: *HAutoMLBench* y *Dataset*.

La clase *HAutoMLBench* es la encargada de interactuar con el usuario, en ella se encuentran todos los métodos que permiten probar los sistemas. Estos métodos son estáticos para evitar crear instancias distintas con el fin de acceder al estado del benchmark y sus funcionalidades. 

La clase *Dataset*, por otro lado, se encarga de modelar cada uno de los conjuntos que pertenecen al benchmark y almacena sus propiedades. Las propiedades de un conjunto son el nombre, url utilizada para la descarga de su contenido, informaciones de la naturaleza de sus datos y una función de carga llamada *loader_function*. Esta función es la que permite acceder a su contenido. 

Esta clase hereda de la clase *YamlAble*. Esto le brinda la posibilidad  de serializarse como un objeto *.yaml*,  permitiendo guardar cada uno de sus valores y acceder a ellos de forma rápida y fácil. 

**Propiedades de un Dataset**

Las propiedades en los conjuntos permiten almacenar información relevante para su utilización. El nombre, la url y su función de carga posibilitan acceder a su contenido.

La información sobre sus metadatos permite identificar la complejidad de la tarea que resuelve.

Estos metadatos contienen el número de columnas, el número de instancias de sus partes: entrenamiento y prueba, la tarea que resuelven y el número de valores null que poseen. Además, se brinda para cada conjunto el tipo semántico inferido para cada una de sus columnas. Esta inferencia se realiza a partir de la interpretación de la descripción del conjunto original. El nombre de la etiqueta de salida es otro de los metadatos, para el conjunto *google-guest* se guarda una lista, ya que posee varias etiquetas que pueden utilizarse como salida. 

En las tareas de clasificación se tiene una lista con los distintos valores de la etiqueta a predecir. En la binaria se tiene una propiedad *positive_class* que indica la clase positiva. Por último se tiene un metadato para el número de clases en las tareas de clasificación y el balance de las mismas. 

El balance se mide por el número de instancias de la clase minoritaria frente a la mayoritaria en el conjunto de entrenamiento. 

```
class Dataset:
    url: str
    name: str
    loader_function : function
    info : dict = {'n_instances': list[int],
                    'n_columns': int , 
                    'columns_types': dict{'name' : type},
                    'targets': list[str] or str,
                    'null_values': int,
                    'task': str 
                    'positive_class': Any,
                    'class_labels': list[Any] 
                    'n_classes': int, 
                    'class_balance':float}
```

**Tipos semánticos**: 

Los tipos semánticos de cada columna aportan información sobre el dominio al que pertenecen los datos. El brindarlos como información garantiza que el conjunto es procesado respetando el significado de cada una de las entradas. Estos se dividen atendiendo a los diferentes tipo de datos y se agregan unas clasificaciones especiales para ayudar a entender aún más el problema.

La estructura de los conjuntos es independiente de los tipos semánticos de sus columnas. Los usuarios pueden decidir si usar los tipos o no, en la evaluación. 

Para las entradas de texto :  

1. text: Posee más de una idea. Estas pueden o no tener relación.

2. sentence: Es un texto que transmite una idea esencial que para su entendimiento no puede ser separado en partes.

3. word:  Tiene significado fijo, es la más pequeña entidad.

4. syntagma: Es considerado una estructura más compleja que word ya que puede incluir varias de estas. Es una frase que debe procesarse de forma conjunta porque pierde su significado. Esta no indica acción ni estado por tanto no se considera oración. 

Para las entradas numéricas:

1. int: numero entero.

2. float: numero decimal.

Variables categóricas:

1. category: La variable puede ser dividida en tipos o categorías.

Variables booleanas:

1. boolean: Se tomaron solo como variables booleanas aquellas que tienen valores True o False.

Variables de tiempo :

1. datetime: Columnas que especifican una fecha o un tiempo.

Entradas Imagen:  

1. image: Una entidad imagen

2. path_image: el enlace a una imagen o el path de una carpeta de imágenes.

Variables relacionadas a reconocimiento de entidades:

1. SeqTokens : una lista de palabras a etiquetar como entidad .

2. SeqLabels: una lista de etiquetas que corresponden al formato IOB2. 

**Function_loader **

La clase *Dataset* posee un método para descargar, abrir y leer el contenido propio de cada una de sus instancias. Este método es abstracto, por lo que cada conjunto de datos debe realizar su propia implementación para realizar las tareas antes descritas. 

En el caso de los conjuntos ya pertenecientes al benchmark sus implementaciones poseen ciertos parámetros de entrada para describir las características del contenido que retornan. Es posible especificar el formato de salida del contenido mediante el parámetro *format*, este corresponde con el tipo lista o dataframe de pandas. 

El parámetro *samples* indica si el conjunto se divide en dos: entrenamiento y prueba, o se retorna completo. 

También es posible separar la columna de salida de las de entrada mediante el parámetro booleano *in**\_**x**\_**y*. 

Cada conjunto conoce su etiqueta a predecir. El conjunto *google-guest*posee otras que pueden utilizarse como salida, por lo que su método admite un parámetro *target*.  

```
def function_loader(self, format = "pandas", in_x_y = False ,samples =  2)
    '''
    input: self: Dataset
        format: str
        in_x_y : bool
        samples : int
        target: str only when dataset matches google-guest
    output: tuple	   
        if in_x_y = True, samples = 2  return X_train, y_train ,X_test, y_test
        in_x_y = True, samples = 1 return X_all, y_all
        in_x_y = False, samples = 2  return train, test
        format = "pandas", in_x_y = False, samples =  1  return all
            format:
            pandas : table 
            list : list [[intance1],[intance2]....]
    '''
```



### Métodos de interacción del usuario

**Método init**

El método *init* inicializa el benchmark. Para ello almacena en un archivo, de todos los conjuntos que posee, los nombres, urls, las informaciones y los nombres de las funciones de carga. Luego construye y serializa las instancias de los conjuntos para su uso futuro. Es importante que este método solo se ejecute una sola vez, debido a que los cambios que se realicen con anterioridad se perderán. El benchmark vuelve a su estado inicial por defecto, con sus conjuntos originales.

**Método get_dataset**

Una vez se inicializa el benchmark para evaluar los sistemas se debe tener acceso a los conjuntos de datos.

El método *get**\_**dataset* a partir de un nombre obtiene la instancia de ese conjunto almacenado.

A tener la instancia se puede acceder a cada una de sus propiedades, su nombre, sus metadatos, url y su función de carga.  

Consultar el ejemplo:  para obtener más información respecto a la estructura del método. 

```
def get_dataset(name):
    '''
    Input: name: str 
    Output: dataset : Dataset
    ''' 
```

**Método add_dataset**

El método *add**\_**dataset* agrega un nuevo conjunto al benchmark. 

Recibe como entrada el nombre, la url, la función de carga y su parámetro opcional, los metadatos. Si se introducen los metadatos estos pasan por un proceso de verificación con el fin de validar que se introduzcan correctamente. Sus campos son obligatorios, si no se tiene información respecto a uno se deben marcar como *None*.

Se debe tener en cuenta que si existe un conjunto con el mismo nombre se remplazará. 

Todos los errores se imprimen en pantalla.

Parámetros por defecto : 

```
def add_dataset(cls, name, url, function, metadata = None ):
        '''
        Input:
            name: str
            url: str
            function : function
            metadata : dict{'n_instances': list[int],
                            'n_columns': int , 
                            'columns_types': dict{'name': type},
                            'targets': list[str] or str,
                            'null_values': int,
                            'task': str 
                            'positive_class': Any,
                            'class_labels': list[Any] 
                            'n_classes': int, 
                            'class_balance': float}
        Output: 
        ''' 
```

**Método remove_dataset**

Este método, como su nombre lo indica, remueve el conjunto que coincida con el nombre que se introduce como parámetro de entrada. 

Produce error en caso de que exista algún problema durante la escritura de los archivos del estado del benchmark.

  Parámetros por defecto : 

```
def remove_dataset(name):
    '''
    input: name: str
    output: None
    '''
```

**Método filter**

El objetivo del método *filter* es devolver una lista con los nombres de los conjuntos de datos que cumpla con las restricciones de los parámetros de entrada. La primera restricción que puede ser introducida es el tipo de tarea. El segundo parámetro es el cumplimiento de una propiedad numérica del tipo Dataset, esta propiedad debe ser característica de los metadatos. En caso de no establecer ninguna restricción, se devuelve una lista con todos los conjuntos introducidos hasta el momento.

```
def filter(cls, task = None,expresion = None):
    '''
    input:
        task: str = 'binary', 'regression, 'multiclass', 'multiclass'
        expression: tuple(len(3)) = (property,min,max) : min <= property < max
        property: str = 'n_instances','n_columns', 'null_values','n_classes','class_balance'
    output: list[str]
    '''  
```

**Método evaluate** 

El método *evaluate* mide el rendimiento de una herramienta de aprendizaje automático. 

Posee parámetros obligatorios como el nombre del conjunto, la lista de valores verdaderos de la salida y los valores predichos.

Ambas secuencias deben coincidir en tipos y en tamaño. En el caso de que el conjunto pertenezca al benchmark y que contenga sus metadatos correctamente etiquetados se puede prescindir de introducir los parámetros opcionales. 

Los parámetros opcionales corresponden a la tarea que se pretende evaluar, los distintos valores de la etiqueta de salida, este último solo es necesario para evaluar tareas de clasificación multiclase. Otro parámetro opcional es la etiqueta de la clase positiva, es necesaria para clasificaciones binarias y cuando la salida tiene valores no numéricos. 

Los últimos parámetros coinciden con el nombre del archivo y la ruta en donde se quieren guardar los resultados. 

El método utiliza las métricas implementadas en el paquete de python *sklearn*. Según el tipo de tarea se evalúan sus métricas correspondientes. Para las métricas de entidades y relaciones se provee una Implementación de la *precisión*, *recobrado* y *f1**\_**beta*. 

La salida incluye cada una de las métricas y el valor obtenido para cada una de ellas. Durante la ejecución se escribe un archivo *json* con los resultados obtenidos. Este archivo es capaz de almacenar todos los resultados de un sistema en todos los conjuntos.

El archivo tiene la forma de: 

```
dict{key: name_datasets, value: {key: name_metric,value: value_metric}}
```

```
def evaluate( cls, name, y_true, y_pred, is_multilabel = False, task =None, positive_class = None , class_labels = None, save_path = None, name_archive ='results'):
    '''
    input: name: str
           y_true: array, list
           y_pred: array, list
           Non-mandatory parameters input:
           task: str
           positive_class: str
           class_labels: list[str]
           save_path: str : path
           name_archive: str
           output: 
           result: {key: name_metric,value: value_metric}
           '''
```

 ### Métodos auxiliares 

En el archivo *utils.py* se tienen los métodos auxiliares que se encargan de definir las tareas de la clase *Dataset*. Ejemplo de estas funcionalidades son la capacidad de serializarse y el proceso inverso. El archivo también guarda métodos que ayudan a la clase *HAutoMLBench*, aquí se almacenan los métodos de  inicialización y cambio de sus variables de estado: nombres de los conjuntos que pertenecen, sus url, los nombres de sus funciones de carga y sus metadatos.  