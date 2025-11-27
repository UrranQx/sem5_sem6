# Java CNN for MNIST Classification using Java Vector API

Реализация сверточной нейронной сети (CNN) на Java с использованием Java Vector API для классификации цифр MNIST.

## Структура проекта

```
java_cnn/
├── src/main/java/cnn/
│   ├── Main.java              # Главный файл с инициализацией, обучением и тестированием
│   ├── CNN.java               # Класс нейронной сети
│   ├── layers/                # Слои нейронной сети (с Vector API оптимизацией)
│   │   ├── Layer.java         # Интерфейс слоя
│   │   ├── ConvLayer.java     # Сверточный слой (векторизованный)
│   │   ├── MaxPoolLayer.java  # Слой макс-пулинга (векторизованный)
│   │   ├── ReLULayer.java     # Функция активации ReLU (векторизованная)
│   │   ├── FlattenLayer.java  # Слой выравнивания
│   │   ├── DenseLayer.java    # Полносвязный слой
│   │   └── SoftmaxLayer.java  # Softmax с кросс-энтропией
│   ├── utils/                 # Утилиты
│   │   ├── VectorOps.java     # Векторизованные операции (Vector API)
│   │   ├── Tensor3D.java      # 3D тензор для изображений (с Vector API)
│   │   └── ConfusionMatrix.java # Матрица ошибок
│   ├── training/              # Классы для обучения (NEW!)
│   │   ├── BatchTrainer.java  # Mini-batch обучение с многопоточностью
│   │   └── ParallelUtils.java # Утилиты для параллельных вычислений
│   ├── benchmark/             # Бенчмарки для сравнения производительности
│   │   ├── VectorBenchmark.java # Сравнение Vector API vs Scalar
│   │   └── ScalarOps.java     # Скалярные (не-векторизованные) операции
│   └── data/
│       └── MNISTLoader.java   # Загрузчик данных MNIST
└── README.md
```

## Архитектура сети

```
Input: 28x28x1 (изображение MNIST)
    ↓
Conv2D: 8 фильтров, 3x3, padding=1 → 28x28x8
    ↓
ReLU
    ↓
MaxPool: 2x2 → 14x14x8
    ↓
Conv2D: 16 фильтров, 3x3, padding=1 → 14x14x16
    ↓
ReLU
    ↓
MaxPool: 2x2 → 7x7x16
    ↓
Flatten → 784
    ↓
Dense: 128
    ↓
ReLU
    ↓
Dense: 10
    ↓
Softmax → вероятности классов 0-9
```

## Требования

- Java 17+ (для поддержки Java Vector API)
- Модуль `jdk.incubator.vector`

## Компиляция

```bash
cd java_cnn
mkdir -p build
javac --add-modules jdk.incubator.vector -d build $(find src -name "*.java")
```
Или
```bash
javac --add-modules jdk.incubator.vector -d build src/main/java/cnn/*.java src/main/java/cnn/layers/*.java src/main/java/cnn/utils/*.java src/main/java/cnn/data/*.java src/main/java/cnn/benchmark/*.java
```

## Запуск

```bash
java --add-modules jdk.incubator.vector -cp build cnn.Main
```

## Запуск бенчмарка (сравнение Vector API vs Scalar)

```bash
java --add-modules jdk.incubator.vector -cp build cnn.benchmark.VectorBenchmark
```

## Использование реальных данных MNIST

Для использования настоящего датасета MNIST, скачайте следующие файлы в директорию `mnist_data/`:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`
Или если используете EMNIST, то для большого набора цифр:
- `emnist-digits-train-images-idx3-ubyte.gz`
- `emnist-digits-train-labels-idx1-ubyte.gz`
- `emnist-digits-test-images-idx3-ubyte.gz`
- `emnist-digits-test-labels-idx1-ubyte.gz`
Малого:
- `emnist-mnist-train-images-idx3-ubyte.gz`
- `emnist-mnist-train-labels-idx1-ubyte.gz`
- `emnist-mnist-test-images-idx3-ubyte.gz`
- `emnist-mnist-test-labels-idx1-ubyte.gz`

Файлы можно скачать с: http://yann.lecun.com/exdb/mnist/
Или с https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip

Если файлы не найдены, программа автоматически генерирует синтетические данные для тестирования.

## Параметры обучения

В файле `Main.java` можно изменить:

```java
private static final float LEARNING_RATE = 0.001f;  // Скорость обучения
private static final int EPOCHS = 3;                // Количество эпох
private static final double TRAIN_RATIO = 0.7;      // Соотношение обучение/тест (70%/30%)

// Параметры batch-обучения и многопоточности
private static final int BATCH_SIZE = 32;           // Размер мини-батча
private static final boolean USE_BATCH_TRAINING = true;  // Включить/выключить batch-обучение
private static final boolean USE_PARALLEL_EVAL = true;   // Включить/выключить параллельную оценку
```

## Mini-Batch обучение (NEW!)

Mini-batch обучение позволяет обрабатывать несколько образцов за один шаг обновления весов:

### Преимущества batch-обучения:
1. **Более стабильные градиенты** - усреднение по батчу уменьшает шум
2. **Лучшее использование аппаратных ресурсов** - векторизованные операции на батчах
3. **Возможность параллельной обработки** - разные образцы могут обрабатываться параллельно
4. **Улучшенная сходимость** - во многих случаях batch-обучение сходится быстрее

### Использование BatchTrainer:
```java
BatchTrainer trainer = new BatchTrainer(cnn, learningRate, batchSize);
// или с явным указанием числа потоков:
BatchTrainer trainer = new BatchTrainer(cnn, learningRate, batchSize, numThreads);

// Обучение одной эпохи
BatchTrainer.TrainingResult result = trainer.trainEpoch(trainX, trainY, trainLabels, seed);
System.out.println("Loss: " + result.loss + ", Accuracy: " + result.accuracy);

// Параллельное предсказание
int[] predictions = trainer.parallelPredict(testX, testLabels);

// Не забудьте закрыть тренер
trainer.shutdown();
```

## Многопоточность и параллельные вычисления (NEW!)

### ParallelUtils - утилиты для параллельных операций:

```java
// Параллельное предсказание с использованием ForkJoinPool
int[] predictions = ParallelUtils.parallelPredict(cnn, inputs);

// Параллельная оценка с построением confusion matrix
ConfusionMatrix matrix = ParallelUtils.parallelEvaluate(cnn, testX, testLabels, 10);

// Параллельное вычисление accuracy
float accuracy = ParallelUtils.parallelAccuracy(predictions, labels);

// Предсказание с явным контролем числа потоков
int[] predictions = ParallelUtils.chunkedParallelPredict(cnn, inputs, numThreads);

// Бенчмарк сравнения последовательной и параллельной обработки
String benchmarkResults = ParallelUtils.benchmarkPrediction(cnn, testX);
```

### Пример вывода бенчмарка параллельного предсказания:
```
==================================================
Prediction Benchmark (2100 samples)
==================================================
Sequential: 1234 ms (1701.46 samples/sec)
Parallel:   456 ms (4605.26 samples/sec)
Speedup:    2.71x
Results match: Yes
==================================================
```

### Рекомендации по использованию многопоточности:

1. **Параллельное предсказание** - безопасно использовать, т.к. forward pass независим
2. **Параллельное обучение** - требует осторожности из-за состояния слоев
3. **Оптимальное число потоков** - обычно равно числу ядер CPU
4. **Синхронизация** - используется для thread-safety при доступе к CNN

## Использование Java Vector API

Java Vector API используется для ускорения вычислений во всех ключевых операциях CNN:

### Оптимизированные слои:

1. **ConvLayer** - сверточный слой с SIMD-оптимизацией:
   - Прямой проход использует векторизованные операции для stride=1
   - Обратный проход с векторизованным обновлением весов

2. **MaxPoolLayer** - слой макс-пулинга:
   - Векторизованный поиск максимума для больших пулов

3. **ReLULayer** - функция активации:
   - Векторизованный ReLU для 1D массивов и 3D тензоров
   - Векторизованный обратный проход

4. **Tensor3D** - 3D тензор:
   - Векторизованные операции: flatten, fromFlattened, relu, copy, pad, add, scale

### Класс VectorOps

Содержит все векторизованные операции:

```java
// Базовые операции
VectorOps.add(a, b)           // Сложение массивов
VectorOps.subtract(a, b)      // Вычитание массивов
VectorOps.multiply(a, b)      // Поэлементное умножение
VectorOps.scale(a, scalar)    // Умножение на скаляр
VectorOps.dot(a, b)           // Скалярное произведение

// Активации
VectorOps.relu(a)             // ReLU активация
VectorOps.softmax(a)          // Softmax активация
VectorOps.relu3D(input, output)  // ReLU для 3D тензоров

// Операции с матрицами
VectorOps.matVecMul(matrix, vec)  // Матрично-векторное умножение
VectorOps.transpose(matrix)       // Транспонирование

// Операции свертки и пулинга
VectorOps.convolveFilter(...)     // Векторизованная свертка
VectorOps.maxPool2D(...)          // Векторизованный макс-пулинг

// In-place операции (для экономии памяти)
VectorOps.addInPlace(a, b)
VectorOps.scaleInPlace(a, scalar)
VectorOps.reluInPlace(a)
```

### Пример использования Vector API:
```java
private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

public static float[] add(float[] a, float[] b) {
    float[] result = new float[a.length];
    int upperBound = SPECIES.loopBound(a.length);
    
    for (int i = 0; i < upperBound; i += SPECIES.length()) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        va.add(vb).intoArray(result, i);
    }
    // Обработка остатка...
    return result;
}
```

## Результаты бенчмарков

Сравнение производительности Vector API vs скалярных операций (AVX2, 8 floats per vector):

| Операция | Размер | Scalar | Vector | Ускорение |
|----------|--------|--------|--------|-----------|
| Dot Product | 4096 | 3.83ms | 0.49ms | **7.77x** |
| Dot Product | 65536 | 61.49ms | 7.96ms | **7.73x** |
| MatVec 128x784 | 100352 | 88.27ms | 9.80ms | **9.00x** |
| MatVec 1024x1024 | 1048576 | 944.36ms | 103.19ms | **9.15x** |
| Conv 16x32x32 k=3 | 16384 | 18.62ms | 11.42ms | **1.63x** |
| Conv 32x64x64 k=3 | 131072 | 154.63ms | 95.59ms | **1.62x** |
| ReLU3D 64x64x64 | 262144 | 16.57ms | 9.44ms | **1.76x** |

**Ключевые результаты:**
- Матрично-векторное умножение: **~9x ускорение** (критично для Dense слоёв)
- Скалярное произведение: **~7-8x ускорение** (основа матричных операций)
- Свертка: **~1.6x ускорение** для больших входных данных
- 3D ReLU: **~1.8x ускорение** для больших тензоров

> **Примечание:** Для небольших массивов (< 256 элементов) накладные расходы на векторизацию могут превышать выигрыш. Основной выигрыш достигается на больших данных.

## Вывод программы

1. Загрузка данных
2. Разделение на обучающую/тестовую выборки (70%/30%)
3. Архитектура сети
4. Процесс обучения с метриками loss и accuracy
5. **Confusion Matrix** с метриками:
   - Precision (точность)
   - Recall (полнота)
   - F1-Score
6. **10 примеров** с ASCII-визуализацией и предсказаниями
7. **Бенчмарк параллельного предсказания** (сравнение скорости)

## Дополнительные способы оптимизации (потенциальные улучшения)

### Реализованные оптимизации:
- ✅ Java Vector API (SIMD) для всех основных операций
- ✅ Mini-batch обучение (BatchTrainer)
- ✅ Параллельное предсказание (ParallelUtils)
- ✅ Параллельная оценка с confusion matrix

### Возможные будущие улучшения:
1. **GPU-ускорение** - использование OpenCL/CUDA через JNI или TornadoVM
2. **Data augmentation** - увеличение датасета путем трансформаций
3. **Learning rate scheduling** - динамическое изменение скорости обучения
4. **Dropout слои** - регуляризация для предотвращения переобучения
5. **Batch Normalization** - нормализация активаций между слоями
6. **Adam/RMSprop оптимизаторы** - вместо простого SGD
7. **Распределенное обучение** - обучение на нескольких машинах

## Автор

Создано для учебного проекта по реализации CNN на Java.
