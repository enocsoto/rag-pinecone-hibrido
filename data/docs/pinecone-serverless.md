# Índices serverless y namespaces en Pinecone

Un índice serverless en Pinecone no reserva pods fijos. La plataforma crea el índice en una nube y una región, por ejemplo aws y us-east-1, y cobra por uso. Para embeddings de OpenAI text-embedding-3-small la dimensión del índice tiene que ser 1536 y la métrica habitual es cosine. Si el índice se creó con 512 o 768, el upsert de un vector de 1536 falla por mismatch de dimensiones. Hay que borrar ese índice o usar otro nombre.

El nombre del índice solo admite minúsculas, números y guiones. ServerlessSpec fija cloud y region en el momento de crear. Después de create_index el estado pasa por Initializing. No se debe hacer upsert hasta que status.ready sea verdadero. describe_index_stats puede mostrar total_vector_count en cero unos segundos después de un upsert: las estadísticas son eventualmente consistentes.

## Namespace como aislamiento

Un namespace es una partición lógica dentro del mismo índice. La búsqueda que no indica namespace mezcla documentos de todos los inquilinos y se vuelve ruidosa. En este módulo el namespace por defecto se llama tecnico. La ingesta y la consulta tienen que usar el mismo valor. Un segundo conjunto de documentos, por ejemplo legales, iría a otro namespace sin recrear el índice.

Cada vector lleva id estable, values y metadata. El texto del chunk vive en la metadata, bajo la clave text, para no ir a buscar el párrafo a una base relacional. También se guardan source (el archivo de origen), page (el número de chunk, porque estos documentos son Markdown y no tienen página de PDF) y category.

El lote de upsert no debería superar unas decenas de vectores. Un lote enorme choca con el límite de tamaño del request. Este pipeline corta la carga en tandas de 50. Los ids son deterministas: nombre del documento, numeral y el índice del chunk. Repetir la ingesta reemplaza el mismo id en lugar de duplicar el párrafo.

Antes de una reingesta, el pipeline vacía solo el namespace tecnico. No toca otros namespaces del mismo índice. Ese es el aislamiento entre conjuntos de documentos: borrar tecnico no borra un namespace legal que pudiera existir al lado. La clave de API no se imprime ni se escribe en los logs. Si el índice ya existe con otra dimensión, setup_index se detiene y pide otro INDEX_NAME, porque Pinecone no deja cambiar la dimensión de un índice serverless ya creado. La región us-east-1 es la del plan de este ejercicio. Mover el índice a otra región implica crearlo de nuevo y volver a embeber, porque los vectores no se copian solos entre regiones.
