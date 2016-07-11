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

#define LOG_ERROR_(message) fprintf(stderr, "%s:%d:%s(): %s", __FILE__, __LINE__, __FUNCTION__, message); 
#define LOG_WARNING_(message) fprintf(stderr, "%s:%d:%s(): %s", __FILE__, __LINE__, __FUNCTION__, message); 
#define LOG_NOTICE_(message) fprintf(stderr , "%s", message);
#define LOG_OUT(message) fprintf(stderr , "%d\n\r", message);

typedef struct
{
	int* nodestatus;	//  ¿¿¿¿¿<split, internal, terminal>
 	int* depth;			// ¿¿¿¿
 	int* splitid;		// ¿¿¿¿¿split¿¿
 	double* splitvalue; // !<split¿¿¿¿¿split value
 	int* ndstart;		// !<¿¿¿¿¿index¿¿¿¿¿
 	int* ndcount;		// !<¿¿¿¿¿¿¿¿¿
 	double* ndavg;		// !<¿¿¿¿¿¿¿¿
 	//double* vpredict;	// 
 	int* lson;			// ¿¿¿¿¿¿
 	int* rson;			// ¿¿¿¿¿¿
 	int nodesize;		//!<¿¿¿¿¿¿
} gbdt_tree_t; //!< ¿¿¿¿¿¿¿



typedef struct
{
	int tree_num;		//!< ¿¿¿¿¿¿
 	int fea_num;		//!< feature¿¿¿
 	int data_num;		//!< ¿¿¿¿¿¿¿¿
 	int sample_num;		//!< ¿¿¿¿¿¿¿¿
 	int rand_fea_num;	//!< Feature¿¿¿¿¿

 	double shrink;		//!< ¿¿¿

 	int gbdt_min_node_size; //!< ¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿
 	int gbdt_max_depth; //!< ¿¿¿¿¿¿¿¿¿¿¿¿

 	char train_filename[BUFFER_LENGTH]; //!< ¿¿¿¿¿¿¿¿
 	char model_filename[BUFFER_LENGTH]; //!< ¿¿¿¿¿
} gbdt_info_t; //!< ¿¿¿¿¿¿¿¿¿¿

typedef struct
{
	gbdt_tree_t** reg_forest; //!< ¿¿¿¿
	gbdt_info_t info; //!< GBDT ¿¿¿¿¿

	double* feature_average; //!< Feature ¿¿¿¿¿¿¿¿¿
}gbdt_model_t; //!< GBDT ¿¿¿¿¿¿

typedef struct  
{
	int* fea_pool; //!< ¿¿feature ¿¿¿
 	double* fvalue_list; //!< ¿feature i¿¿¿¿¿¿¿ x_i
 	double* fv; //!< ¿¿¿¿¿¿¿buffer¿¿
 	double* y_list; //!< ¿¿¿y¿¿¿
 	int* order_i; //!< ¿¿¿¿¿
} bufset; //!< ¿¿¿¿¿

typedef struct 
{
	int index_b; //!< ¿¿¿¿¿¿¿¿¿
 	int index_e; //!< ¿¿¿¿¿¿¿¿¿
 	int nodenum; //!< ¿¿¿¿¿¿¿¿
 	double nodesum; //!< ¿¿¿¿¿¿¿y¿¿¿¿¿¿¿
 	double critparent; //!< ¿¿¿¿¿¿

} nodeinfo; //!< ¿¿¿¿¿

typedef struct  
{
	int bestid; //!< ¿¿¿¿¿feature ID
 	double bestsplit; //!< ¿¿¿¿¿x¿
 	int pivot; //!< ¿¿¿¿¿¿¿¿¿
 
} splitinfo; //!<¿¿¿¿¿
 
 /*
 * @brief ÔÚÑµÁ·Êı¾İÖĞ±éÀúËæ»ú³éÈ¡µÄFeatureÑ°ÕÒ·Ö¸îµÄ×î¼ÑÎ»ÖÃ
 *
 * @param <IN> gbdt_inf : Ä£ĞÍµÄÅäÖÃĞÅÏ¢½á¹¹Ìå
 * @param <IN> data_set : ÑµÁ·Êı¾İ³Ø
 * @param <IN> x_fea_value : ÑµÁ·Êı¾İÖĞµÄfeatureÖµ
 * @param <IN> y_score : ÑµÁ·Êı¾İÖĞµÄËùÓĞÄ¿±êÖµ
 * @param <IN> ninf : ¸ù½ÚµãĞÅÏ¢
 * @param <IN> index : ÅÅĞòµÄĞòºÅ
 * @param <IN> spinf : ·ÖÁÑµÄ½ÚµãĞÅÏ¢
 * @return=-1 : ·ÖÁÑÊ§°Ü
 * @return=1 : ·ÖÁÑµÄÌØÊâÇé¿ö£¬ÎŞ·¨Ñ¡³ö¿ÉÒÔ·Ö¸îµÄFeature
 * @return=0 : ·ÖÁÑ³É¹¦
 * */
 int gbdt_tree_node_split(gbdt_info_t gbdt_inf, bufset* data_set, double *x_fea_value, double *y_score,
     nodeinfo ninf, int* index, splitinfo* spinf);
 
 /*
 * @brief Éú³ÉÒ»¿Ã»Ø¹éÊ÷
 *
 * @param <IN> x_fea_value : ÑµÁ·Êı¾İÖĞµÄfeatureÖµ
 * @param <IN> y_gradient : ÑµÁ·Êı¾İÖĞµÄËùÓĞÄ¿±êÖµ
 * @param <IN> gbdt_inf : ÑµÁ·Ä£ĞÍµÄÅäÖÃ²ÎÊı
 * @param <IN> data_set : ÑµÁ·Êı¾İ³Ø
 * @param <IN> index : ÅÅĞòµÄĞòºÅ
 * @param <IN> gbdt_single_tree : »Ø¹éÊ÷µÄÖ¸Õë
 * @param <IN> nrnodes : Ê÷µÄ½ÚµãÊıÄ¿
 * @return=-1 : ÑµÁ·Ê§°Ü
 * @return=0 : ÑµÁ·³É¹¦
 * */
 
 int gbdt_single_tree_estimation(double *x_fea_value, double *y_gradient, 
   gbdt_info_t gbdt_inf, bufset* data_set, 
   int* index, gbdt_tree_t* gbdt_single_tree, int nrnodes );
 
 /*
 * @brief Ê¹ÓÃGradient Boosting Decision TreeÄ£ĞÍ½øĞĞ»Ø¹éÔ¤²â
 *
 * @param <IN> rf_model : Random ForestÄ£ĞÍ½á¹¹µÄÖ¸Õë
 * @param <IN> x_test: ²âÊÔÊı¾İµÄFeatureÖµ
 * @param <OUT> y_predict : Ä£ĞÍÔ¤²âÖµ
 * @return=-1 : Ô¤²âÊ§°Ü
 * @return=1 : Ô¤²â³É¹¦
 * */
 
 int gbdt_regression_predict(gbdt_model_t* gbdt_model, double *x_test, double& ypredict);
 
 /*
 * @brief ¼ÆËãµ¥¿Ã»Ø¹éÊ÷µÄÔ¤²âÖµ
 *
 * @param <IN> x_test : Ô¤²âÊı¾İµÄFeatureÖµ
 * @param <IN> gbdt_single_tree : µ¥¿Ã¾ö²ßÊ÷µÄÖ¸Õë
 * @param <IN> ypred : »Ø¹éÔ¤²âÖµ
 * @param <IN> shrink : Ñ§Ï°ÂÊ
 * @return=-1 : Ô¤²âÊ§°Ü
 * @return=0 : Ô¤²â³É¹¦
 * */
 
 int gbdt_tree_predict(double *x_test, gbdt_tree_t *gbdt_single_tree, double& ypred, double shrink);
 
 /*
 * @brief ÑµÁ·Gradient Boosting Decision TreeÄ£ĞÍ
 *
 * @param <IN> x_fea_value : ÑµÁ·Êı¾İµÄFeatureÖµ
 * @param <IN> y_result_score : ÑµÁ·Êı¾İµÄÄ¿±êÖµ
 * @param <IN> infbox : ÑµÁ·Ä£ĞÍµÄÅäÖÃ²ÎÊı
 * @return=-1 : ÑµÁ·Ê§°Ü
 * @return=0 : ÑµÁ·³É¹¦
 * */
 
 gbdt_model_t* gbdt_regression_train(double *x_fea_value, double *y_result_score, gbdt_info_t infbox);
 
 /*
 * @brief ´ÓÃüÁîĞĞ½âÎö²ÎÊı
 *
 * @param <IN> infbox : Ä£ĞÍÅäÖÃµÄ½á¹¹Ìå
 * @param <IN> argc : ÃüÁîĞĞ²ÎÊı¸öÊı
 * @param <IN> argv : ÃüÁîĞĞ²ÎÊı
 * @return=-1 : ²ÎÊı¶ÁÈ¡Ê§°Ü
 * @return=0 : ²ÎÊı¶ÁÈ¡³É¹¦
 * */
 
 int read_conf_file(gbdt_info_t& infbox, int argc, char* argv[]);
 /*
 * @brief ÒÔseparator×÷Îª·Ö¸ô·û£¬¶ÔÒ»ĞĞ½øĞĞ·Ö¸ô´æ´¢ÔÚitemsÊı×éÖĞ
 *
 * @param <IN> line : Random ForestÄ£ĞÍ½á¹¹µÄÖ¸Õë
 * @param <IN> items[]: ²âÊÔÊı¾İµÄFeatureÖµ
 * @param <IN> items_num : Ä£ĞÍÔ¤²âµÄ·ÖÀàÀàºÅ
 * @param <IN> separator : Ä£ĞÍÔ¤²âµÄ·ÖÀàÀàºÅ
 * @return : ·Ö¸ô³öÀ´µÄ×Ó×Ö·û´®µÄÊıÁ¿
 * */
 int splitline(string line, string items[], int items_num, const char separator);
 
 /*
 * @brief ½«ÑµÁ·ºÃµÄGradient Boosting Decision TreeÄ£ĞÍ´æ´¢ÔÚÎÄ¼şÖĞ
 *
 * @param <IN> model_filename : ´æ´¢µÄÄ£ĞÍÎÄ¼şÃû
 * @param <IN> gbdt_model: ´ı´æ´¢µÄÄ£ĞÍ½á¹¹ÌåÖ¸Õë
 * @return=-1 : ´æ´¢Ê§°Ü
 * @return=1 : ´æ´¢³É¹¦
 * */
 int gbdt_save_model(gbdt_model_t* gbdt_model, char* model_filename);
 /*
 * @brief ´ÓÎÄ¼şÖĞ¶ÁÈ¡Gradient Boosting Decision TreeµÄÄ£ĞÍ
 *
 * @param <IN> model_file : Ä£ĞÍÎÄ¼şÃû
 * @return=NULL : ¶ÁÈ¡Ê§°Ü
 * @return!=NULL : ¶ÁÈ¡³É¹¦
 * */
 gbdt_model_t* gbdt_load_model(char* model_file);
 
 /*
 * @brief ÊÍ·ÅGradient Boosting Decision TreeÄ£ĞÍ
 *
 * @param <IN> rf_model : gbdt_modelÄ£ĞÍµÄÖ¸Õë
 *
 * */
 
 int free_model(gbdt_model_t*& gbdt_model);
 
 void R_qsort_I(double *v, int *I, int i, int j);

