//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

// 文件名字大小的限制, 也是每个单词的大小限制
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

struct vocab_word {
  long long cn; // 词频
  int *point; // 哈夫曼树链路上各个节点的编码
  char *word, *code, codelen; // code表示其哈夫曼编码
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
// train_words:表示实际要做向量化词的个数，该个数可能和输入不相等，会因为词频太低被去掉(>min_count)
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
// alpha: 学习速率
// sample：下采样的权重, pr = 1 - sqrt(sample/count_of_word)
float alpha = 0.025, starting_alpha, sample = 1e-3;
float *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  float d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (float)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    // 某种抽样的策略
    if (a / (float)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (float)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue; // 回车符 '\r'
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        // 文件末尾，返回特殊的字符串
        strcpy(word, (char *)"</s>");
        return;
      } else {
        continue;
      }
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  // 不知道冲突概率怎么样，应该是0或者非常低，char类型，如果是无符号才255
  // 它将每个字符都通过类似horner法则来获得一个大的hash值，最后round回vocab_hash_size
  for (a = 0; a < strlen(word); a++){
     hash = hash * 257 + word[a];
  }
  hash = hash % vocab_hash_size;
  return hash;
}

// 1. vocab_hash: 存放每个值hash值到vocab中位置的索引, hash值本身就是索引, vocab_hash中存的值是另外
//    一个索引，默认是-1，表示该位置是空
// 2. vocab: 存放每个词

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    // 冲突就继续往下一个位置存
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// 1. vocab_size: 表示当前可以存放vocab的位置
// 2. 空间不足时每次增加1000个词的空间，realloc
// 3. vocab结构体中的cn表示词的频率

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0; // 词频在什么时候赋值？
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) {
    // 开放地址探测，每次探测下一个位置，这个和查找方法对应，SearchVocab
    hash = (hash + 1) % vocab_hash_size;
  }
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// 词频大的在前面
// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  // 会释放多余的空间，只保留vocab_size + 1的空间
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  // code和point是用户哈夫曼编码的吗?
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// 类似于SortVocab, 但是其不会code和point的空间，因为空间已经分配好了

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  // min_reduce和min_count之间的区别是?
  // min_count:用于对输入的预处理，必须>=min_count才会被做向量化
  // min_reduce:用户当词数太大时，丢掉频率最低的词
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else {
    free(vocab[a].word);
  }
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) {
    vocab_hash[a] = -1;
  }
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) {
      hash = (hash + 1) % vocab_hash_size;
    }
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  // 这里的空间分配有点大，但是词的个数本来就不会特别多，预料会很大，词库不会很大
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) {
    count[a] = vocab[a].cn;
  }
  for (a = vocab_size; a < vocab_size * 2; a++) {
    count[a] = 1e15;
  }
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  // 已经按照count(cn)排序过，这里从后面开始找两个最小的就行
  // count[vocab_size..]: 全部是按照从大到小的顺序排列的，且当前没用的位置是pos2
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) {
    vocab_hash[a] = -1;
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  // 第一个位置[0]始终是 </s>，所以排序时会从下标为1的开始
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else {
      vocab[i].cn++;
    }
    // reduce的时机是：超过哈希表容量的 70%，但是每次ReduceVocab比较浪费计算资源
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

// 保存和读取的文件格式不一样：
// 读取：预料库中
// 写出：每个词以及其词频
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) {
    fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  }
  fclose(fo);
}

// 这个函数和`SaveVocab` 是对应的
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }

  // 为什么将train_file的文件指针移动至末尾?
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  // posix_memalign(void **memptr, size_t alignment, size_t size)
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(float));
  if (syn0 == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  if (hs) {
    // hierachi softmax
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn1 == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++) {
      for (b = 0; b < layer1_size; b++) {
        syn1[a * layer1_size + b] = 0;
      }
    }
  }
  if (negative>0) {
    // negative sampling
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn1neg == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++) {
      for (b = 0; b < layer1_size; b++) {
        syn1neg[a * layer1_size + b] = 0;
      }
    }
  }
  for (a = 0; a < vocab_size; a++) {
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
    }
  }
  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  float f, g;
  clock_t now;
  // 隐藏层神经元向量
  float *neu1 = (float *)calloc(layer1_size, sizeof(float));
  // 隐藏层神经元误差向量
  float *neu1e = (float *)calloc(layer1_size, sizeof(float));
  FILE *fi = fopen(train_file, "rb");
  // 存在分割词的风险，但是即使破坏了一个词，这个不存在的词有两种可能：
  // 1. 完全不受影响，相当于一个词只丢失了一个样本，不至于这么悲剧，刚搞num_threads个词都被丢了？
  // 2. 由于是不存在的词，其由于min_count被丢弃
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
               word_count_actual / (float)(iter * train_words + 1) * 100,
               word_count_actual / ((float)(now - start + 1) / (float)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }

      // 调整学习速率，有个最低的学习速率：starting_alpha * 0.0001
      // 调整的规则=
      alpha = starting_alpha * (1 - word_count_actual / (float)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) {
        alpha = starting_alpha * 0.0001;
      }
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          float ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (float)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
      // 收集MAX_SENTENCE_LENGTH个word，然后从sentence_position=0的位置开始做什么？
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      // 当前线程训练结束则跳出该线程的thread循环
      // 否则重置所有的东西，继续训练，默认训练5次
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) {
      neu1[c] = 0;
    }
    for (c = 0; c < layer1_size; c++) {
      neu1e[c] = 0;
    }
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) {
        if (a != window) {
          c = sentence_position - window + a;
          // 只在当前MAX_SENTENCE_LENGTH长度里面查找窗口内的单词，左边或者右边都会被忽略
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) {
            // 直接将窗口内向量相加
            neu1[c] += syn0[c + last_word * layer1_size];
          }
          cw++;
        }
      }
      if (cw) {
        for (c = 0; c < layer1_size; c++) {
          // 将神经元的结果average
          neu1[c] /= cw;
        }
        if (hs) {
          for (d = 0; d < vocab[word].codelen; d++) {
            f = 0;
            // l2表示当前词路径上的词在syn1上的位置
            l2 = vocab[word].point[d] * layer1_size;
            // Propagate hidden -> output
            for (c = 0; c < layer1_size; c++) {
              f += neu1[c] * syn1[c + l2];
            }
            if (f <= -MAX_EXP) continue;
            else if (f >= MAX_EXP) continue;
            else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            // 'g' is the gradient multiplied by the learning rate
            g = (1 - vocab[word].code[d] - f) * alpha;
            // Propagate errors output -> hidden
            for (c = 0; c < layer1_size; c++) {
              neu1e[c] += g * syn1[c + l2];
            }
            // Learn weights hidden -> output
            for (c = 0; c < layer1_size; c++) {
              syn1[c + l2] += g * neu1[c];
            }
          }
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          // 会抽样坏的单词出来
          // 这个随机抽样出来的结果会去更新neu1e，误差向量
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) {
              target = next_random % (vocab_size - 1) + 1;
            }
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) {
            f += neu1[c] * syn1neg[c + l2];
          }
          if (f > MAX_EXP) {
            g = (label - 1) * alpha;
          } else if (f < -MAX_EXP) {
            g = (label - 0) * alpha;
          } else {
            g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          }
          for (c = 0; c < layer1_size; c++) {
            neu1e[c] += g * syn1neg[c + l2];
          }
          for (c = 0; c < layer1_size; c++) {
            syn1neg[c + l2] += g * neu1[c];
          }
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) {
            syn0[c + last_word * layer1_size] += neu1e[c];
          }
        }
      }
    } else {  //train skip-gram
      // 像是对每个次都做一遍，而cbow，是将window内部的词sum后再做，误差传播
      for (a = b; a < window * 2 + 1 - b; a++) {
        if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          l1 = last_word * layer1_size;
          for (c = 0; c < layer1_size; c++) {
            neu1e[c] = 0;
          }
          // HIERARCHICAL SOFTMAX
          if (hs) {
            for (d = 0; d < vocab[word].codelen; d++) {
              f = 0;
              l2 = vocab[word].point[d] * layer1_size;
              // Propagate hidden -> output
              for (c = 0; c < layer1_size; c++) {
                f += syn0[c + l1] * syn1[c + l2];
              }
              if (f <= -MAX_EXP) continue;
              else if (f >= MAX_EXP) continue;
              else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
              // 'g' is the gradient multiplied by the learning rate
              g = (1 - vocab[word].code[d] - f) * alpha;
              // Propagate errors output -> hidden
              for (c = 0; c < layer1_size; c++) {
                neu1e[c] += g * syn1[c + l2];
              }
              // Learn weights hidden -> output
              for (c = 0; c < layer1_size; c++) {
                syn1[c + l2] += g * syn0[c + l1];
              }
            }
          }
          // NEGATIVE SAMPLING
          if (negative > 0) {
            for (d = 0; d < negative + 1; d++) {
              if (d == 0) {
                target = word;
                label = 1;
              } else {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == 0) {
                  target = next_random % (vocab_size - 1) + 1;
                }
                if (target == word) continue;
                label = 0;
              }
              l2 = target * layer1_size;
              f = 0;
              for (c = 0; c < layer1_size; c++) {
                f += syn0[c + l1] * syn1neg[c + l2];
              }
              if (f > MAX_EXP){
                g = (label - 1) * alpha;
              } else if (f < -MAX_EXP) {
                g = (label - 0) * alpha;
              } else {
                g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
              }
              for (c = 0; c < layer1_size; c++) {
                neu1e[c] += g * syn1neg[c + l2];
              }
              for (c = 0; c < layer1_size; c++) {
                syn1neg[c + l2] += g * syn0[c + l1];
              }
            }
          }
          // Learn weights input -> hidden
          for (c = 0; c < layer1_size; c++) {
            syn0[c + l1] += neu1e[c];
          }
        }
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) {
    ReadVocab();
  } else {
    LearnVocabFromTrainFile();
  }
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) {
    pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  }
  for (a = 0; a < num_threads; a++) {
    pthread_join(pt[a], NULL);
  }
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) {
        for (b = 0; b < layer1_size; b++) {
          fwrite(&syn0[a * layer1_size + b], sizeof(float), 1, fo);
        }
      } else {
        for (b = 0; b < layer1_size; b++) {
          fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
        }
      }
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    float closev, x;
    float *cent = (float *)calloc(classes * layer1_size, sizeof(float));
    for (a = 0; a < vocab_size; a++) {
      cl[a] = a % clcn;
    }
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) {
        cent[b] = 0;
      }
      for (b = 0; b < clcn; b++) {
        centcn[b] = 1;
      }
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) {
          cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        }
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= closev;
        }
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) {
            x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          }
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    }
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) {
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
