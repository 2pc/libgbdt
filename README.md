# libgbdt

Non original ! This code  imported  from other place,  and  only a little modify, some annotation

A implement of LS_Boost Algorithm,mentioned in [Greedy Function Approximation: A Gradient Boosting Machine.](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)


```
make clean 
make all
cd output/test
./gbdt-train -r 0.8 -t 100 -s 0.03 -n 30 -d 5 -m test.model -f ../../train
./gbdt-test ./test.model ../../train
```

参数

```
 	{"help", 0, NULL, 'h'},
 	{"sample_feature_ratio", 1, NULL, 'r'},
 	{"tree_num", 1, NULL, 't'},
 	{"shrink", 1, NULL, 's'},
 	{"min_node_size", 1, NULL, 'n'},
 	{"max_depth", 1, NULL, 'd'},
 	{"model_out", 1, NULL, 'm'},
 	{"train_file", 1, NULL, 'f'},
 	{NULL, 0, NULL, 0}};
```

结构体函数定义  --gradient_boosting.h

```
#include "stdio.h"
#include "string.h"
#include "memory.h"
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include <getopt.h>

#include <string>
using namespace std;
 

#define GBDT_TERMINAL -1
#define GBDT_TOSPLIT  -2
#define GBDT_INTERIOR -3

//#define DEBUG

#define uint32 unsigned int
#define swap_int(a, b) ((a ^= b), (b ^= a), (a ^= b))

#define SAMPLE_TYPE 1
#define SAMPLE_RATIO 1.0

#define BUFFER_LENGTH 10240

#define NO_VALUE 0x7FFFFFFF

#define LOG_ERROR_(message) fprintf(stderr, "%s:%d:%s(): %s", __FILE__, __LINE__, __FUNCTION__, message); ///@brief 输出错误信息
#define LOG_WARNING_(message) fprintf(stderr, "%s:%d:%s(): %s", __FILE__, __LINE__, __FUNCTION__, message); ///@brief 输出警告信息
#define LOG_NOTICE_(message) fprintf(stderr , "%s", message);
#define LOG_OUT(message) fprintf(stderr , "%d", message);

typedef struct
{
	int* nodestatus; //!< 节点状态 待split, 中间节点, 终结点
 	int* depth; // 节点深度
 	int* splitid; //!< 节点对应的split特征
 	double* splitvalue; //!< split特征对应的split value
 	int* ndstart; //!< 节点对应于 Index 的开始位置
 	int* ndcount; //!< 节点内元素的个数
 	double* ndavg; //!< 节点内元素的均值
 	//double* vpredict;
 	int* lson; //!< 节点的左儿子
 	int* rson; //!< 节点的右儿子
 	int nodesize; //!< 树的节点个数

} gbdt_tree_t; //!< 回归树的结构体

typedef struct
{
	int tree_num; //!< 森林树的个数
 	int fea_num; //!< Feature的数量
 	int data_num; //!< 训练数据的数据量
 	int sample_num; //!< 训练数据的采样量
 	int rand_fea_num; //!< Feature的采样数量

 	double shrink; //!< 学习率

 	int gbdt_min_node_size; //!< 树停止的条件，节点覆盖的最少的数据量
 	int gbdt_max_depth; //!< 树停止的条件，树的深度

 	char train_filename[BUFFER_LENGTH]; //!< 训练样本的文件名
 	char model_filename[BUFFER_LENGTH]; //!< 模型文件名
} gbdt_info_t; //!< 模型配置参数的结构体

typedef struct
{
	gbdt_tree_t** reg_forest; //!< 回归森林
	gbdt_info_t info; //!< GBDT的配置参数

	double* feature_average; //!< Feature在训练数据的平均值
}gbdt_model_t; //!< GBDT模型的结构体

typedef struct  
{
	int* fea_pool; //!< 随机 feature 候选池
 	double* fvalue_list; //!< 以feature i 为拉链的特征值 x_i
 	double* fv; //!< 特征值排序用的buffer版本
 	double* y_list; //!< 回归的y值集合
 	int* order_i; //!< 排序的标号
} bufset; //!< 训练数据池

typedef struct 
{
	int index_b; //!< 节点覆盖数据的开始
 	int index_e; //!< 节点覆盖数据的结束
 	int nodenum; //!< 节点覆盖的数据量
 	double nodesum; //!< 节点覆盖的数据y值的和，回归用
 	double critparent; //!< 分裂的评价值

} nodeinfo; //!< 节点的信息

typedef struct  
{
	int bestid; //!< 分裂使用的Feature ID
 	double bestsplit; //!< 分裂边界的x值
 	int pivot; //!< 分裂边界的数据标号
 
} splitinfo; //!< 分裂的信息
 
 /*
 * @brief 在训练数据中遍历随机抽取的Feature寻找分割的最佳位置
 *
 * @param <IN> gbdt_inf : 模型的配置信息结构体
 * @param <IN> data_set : 训练数据池
 * @param <IN> x_fea_value : 训练数据中的feature值
 * @param <IN> y_score : 训练数据中的所有目标值
 * @param <IN> ninf : 根节点信息
 * @param <IN> index : 排序的序号
 * @param <IN> spinf : 分裂的节点信息
 * @return=-1 : 分裂失败
 * @return=1 : 分裂的特殊情况，无法选出可以分割的Feature
 * @return=0 : 分裂成功
 * */
 int gbdt_tree_node_split(gbdt_info_t gbdt_inf, bufset* data_set, double *x_fea_value, double *y_score,
     nodeinfo ninf, int* index, splitinfo* spinf);
 
 /*
 * @brief 生成一棵回归树
 *
 * @param <IN> x_fea_value : 训练数据中的feature值
 * @param <IN> y_gradient : 训练数据中的所有目标值
 * @param <IN> gbdt_inf : 训练模型的配置参数
 * @param <IN> data_set : 训练数据池
 * @param <IN> index : 排序的序号
 * @param <IN> gbdt_single_tree : 回归树的指针
 * @param <IN> nrnodes : 树的节点数目
 * @return=-1 : 训练失败
 * @return=0 : 训练成功
 * */
 
 int gbdt_single_tree_estimation(double *x_fea_value, double *y_gradient, 
   gbdt_info_t gbdt_inf, bufset* data_set, 
   int* index, gbdt_tree_t* gbdt_single_tree, int nrnodes );
 
 /*
 * @brief 使用Gradient Boosting Decision Tree模型进行回归预测
 *
 * @param <IN> rf_model : Random Forest模型结构的指针
 * @param <IN> x_test: 测试数据的Feature值
 * @param <OUT> y_predict : 模型预测值
 * @return=-1 : 预测失败
 * @return=1 : 预测成功
 * */
 
 int gbdt_regression_predict(gbdt_model_t* gbdt_model, double *x_test, double& ypredict);
 
 /*
 * @brief 计算单棵回归树的预测值
 *
 * @param <IN> x_test : 预测数据的Feature值
 * @param <IN> gbdt_single_tree : 单棵决策树的指针
 * @param <IN> ypred : 回归预测值
 * @param <IN> shrink : 学习率
 * @return=-1 : 预测失败
 * @return=0 : 预测成功
 * */
 
 int gbdt_tree_predict(double *x_test, gbdt_tree_t *gbdt_single_tree, double& ypred, double shrink);
 
 /*
 * @brief 训练Gradient Boosting Decision Tree模型
 *
 * @param <IN> x_fea_value : 训练数据的Feature值
 * @param <IN> y_result_score : 训练数据的目标值
 * @param <IN> infbox : 训练模型的配置参数
 * @return=-1 : 训练失败
 * @return=0 : 训练成功
 * */
 
 gbdt_model_t* gbdt_regression_train(double *x_fea_value, double *y_result_score, gbdt_info_t infbox);
 
 /*
 * @brief 从命令行解析参数
 *
 * @param <IN> infbox : 模型配置的结构体
 * @param <IN> argc : 命令行参数个数
 * @param <IN> argv : 命令行参数
 * @return=-1 : 参数读取失败
 * @return=0 : 参数读取成功
 * */
 
 int read_conf_file(gbdt_info_t& infbox, int argc, char* argv[]);
 /*
 * @brief 以separator作为分隔符，对一行进行分隔存储在items数组中
 *
 * @param <IN> line : Random Forest模型结构的指针
 * @param <IN> items[]: 测试数据的Feature值
 * @param <IN> items_num : 模型预测的分类类号
 * @param <IN> separator : 模型预测的分类类号
 * @return : 分隔出来的子字符串的数量
 * */
 int splitline(string line, string items[], int items_num, const char separator);
 
 /*
 * @brief 将训练好的Gradient Boosting Decision Tree模型存储在文件中
 *
 * @param <IN> model_filename : 存储的模型文件名
 * @param <IN> gbdt_model: 待存储的模型结构体指针
 * @return=-1 : 存储失败
 * @return=1 : 存储成功
 * */
 int gbdt_save_model(gbdt_model_t* gbdt_model, char* model_filename);
 /*
 * @brief 从文件中读取Gradient Boosting Decision Tree的模型
 *
 * @param <IN> model_file : 模型文件名
 * @return=NULL : 读取失败
 * @return!=NULL : 读取成功
 * */
 gbdt_model_t* gbdt_load_model(char* model_file);
 
 /*
 * @brief 释放Gradient Boosting Decision Tree模型
 *
 * @param <IN> rf_model : gbdt_model模型的指针
 *
 * */
 
 int free_model(gbdt_model_t*& gbdt_model);
 
 void R_qsort_I(double *v, int *I, int i, int j);
```
