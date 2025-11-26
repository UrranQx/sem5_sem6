# Java CNN for MNIST Classification using Java Vector API

Реализация сверточной нейронной сети (CNN) на Java с использованием Java Vector API для классификации цифр MNIST.

## Структура проекта

```
java_cnn/
├── src/main/java/cnn/
│   ├── Main.java              # Главный файл с инициализацией, обучением и тестированием
│   ├── CNN.java               # Класс нейронной сети
│   ├── layers/                # Слои нейронной сети
│   │   ├── Layer.java         # Интерфейс слоя
│   │   ├── ConvLayer.java     # Сверточный слой
│   │   ├── MaxPoolLayer.java  # Слой макс-пулинга
│   │   ├── ReLULayer.java     # Функция активации ReLU
│   │   ├── FlattenLayer.java  # Слой выравнивания
│   │   ├── DenseLayer.java    # Полносвязный слой
│   │   └── SoftmaxLayer.java  # Softmax с кросс-энтропией
│   ├── utils/                 # Утилиты
│   │   ├── VectorOps.java     # Векторизованные операции (Vector API)
│   │   ├── Tensor3D.java      # 3D тензор для изображений
│   │   └── ConfusionMatrix.java # Матрица ошибок
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

## Запуск

```bash
java --add-modules jdk.incubator.vector -cp build cnn.Main
```

## Использование реальных данных MNIST

Для использования настоящего датасета MNIST, скачайте следующие файлы в директорию `mnist_data/`:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

Файлы можно скачать с: http://yann.lecun.com/exdb/mnist/

Если файлы не найдены, программа автоматически генерирует синтетические данные для тестирования.

## Параметры обучения

В файле `Main.java` можно изменить:

```java
private static final float LEARNING_RATE = 0.001f;  // Скорость обучения
private static final int EPOCHS = 3;                // Количество эпох
private static final double TRAIN_RATIO = 0.7;      // Соотношение обучение/тест (70%/30%)
```

## Использование Java Vector API

Java Vector API используется для ускорения вычислений в классе `VectorOps.java`:

- Сложение/вычитание/умножение векторов
- Скалярное произведение
- ReLU активация
- Матрично-векторное умножение

Пример использования Vector API:
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

## Автор

Создано для учебного проекта по реализации CNN на Java.
