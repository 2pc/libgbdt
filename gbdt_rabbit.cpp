/**
* @file rf_rabbit.cpp
* @author Daren Li 
* @date Mon Feb  12 18:54:15 CST 2011
* @version 1.0.0 
* @brief 
*  
**/

 #include <stdio.h>
// #include <ul_conf.h>
 #include <string>
 #include <sstream>
 #include <map>
 
 #include "gradient_boosting.h"
 
 using namespace std;
 
 ///@brief interface MACRO define
 #ifdef __cplusplus
 
 #ifndef _SO_API_
 #define _SO_API_(type) extern "C" type
 #endif
 
 #else
 
 #ifndef _SO_API_
 #define _SO_API_(type) type
 #endif
 
 #endif
 
 ///@brief define the buffer size macro
 
 //#define OUTPUT_DEBUG_INFO
 
 ///@brief global sources
 gbdt_model_t *p_gbdt_model = NULL;
 
 /**
  * @brief define the para during rabbit test
  */
 typedef struct thread_para
 {
   char ori[BUFFER_LENGTH];
   double* x_test;
   string* items;
   double y_reg_predict;
 }thread_para_t;
 
 typedef map <string,string> hmap;///@brief the rabbit result type define
 
 /**
  * @brief the main thread init function
  * init the global dict to be used
  * */
/* 
_SO_API_(int) main_thread_init(Ul_confdata * conf)
 {
   int re = 0;
   char buf[BUFFER_LENGTH] = {0};
   
   if(ul_getconfstr(conf, "gbdt_model_name", buf) != 1)
   {
     fprintf(stderr, "get gbdt_model_name config error. -_-\n");
     re = -1;
     goto EXIT;
   }
 
 #ifdef OUTPUT_DEBUG_INFO
   fprintf(stderr, "load model in : '%s'\n", buf);
 #endif
   p_gbdt_model = gbdt_load_model(buf);
   if(!p_gbdt_model)
   {
     fprintf(stderr, "load gbdt model error.\n");
     re = -2;
     goto EXIT;
   }
   
 EXIT:
   return re;
 }
*/
 /**
  * @brief the main thread destroy function
  * free the global max entropy model 
  * 
  * */
 _SO_API_(void) main_thread_des(void * arg)
 {
   if(p_gbdt_model)
   {
     free_model(p_gbdt_model);
   }
 }
 
 /**
   * hte thread resource init function
   * init the parameter to be use in max entropy model test
   * */
  _SO_API_(void) * thread_resource_init(int count , int arg[])
  {
     thread_para_t *p_para = new thread_para_t;
     if(!p_para) 
     {
       return NULL;
     }
     if(p_gbdt_model)
     {
      p_para->x_test = new double[p_gbdt_model->info.fea_num];
      p_para->items = new string[p_gbdt_model->info.fea_num+5];
     }
    
    return p_para;
  }
  
  /**
   * @brief the thread resource destroy function
   * free the parameter the max entropy model testing used
   * */
  _SO_API_(void) thread_resource_des(void * arg)
  {
    if(!arg) 
    {
      return;
    }
    thread_para_t *p_para = (thread_para_t *)arg;
  
    if(p_para->x_test)
   delete[] p_para->x_test;
    if(p_para->items)
     delete[] p_para->items;
    delete p_para;
  }
  
  /**
   * @brief the filter query function
   * */
  _SO_API_(int) filter_query(char * query)
  {
    return 0;
  }
  
  void fill_no_value_aver(gbdt_model_t* gbdt_model, double* x_test)
  {
   for(int i = 0; i < gbdt_model->info.fea_num; i++)
   {
   if(x_test[i] == -1)
   {
   x_test[i] = gbdt_model->feature_average[i];
   }
   }
  }
  
  /**
   * @brief the thread run function
   * do max entropy model testing in this function
   * */
  _SO_API_(void *) thread_run(char* line,void * parg, 
    int count, void *rarg[], 
    int * status, struct timeval* t3)
  {
   thread_para_t *p_para = (thread_para_t *)parg;
  
   if(p_gbdt_model == NULL)
   {
   return p_para;
   }
  
   int it;
   char feat[BUFFER_LENGTH];
  
   double value = 0;
   double realv = 0;
  
   for(int i = 0; i < p_gbdt_model->info.fea_num; i++)
   {
   p_para->x_test[i] = -1;
   }
  
   strncpy(p_para->ori, line, BUFFER_LENGTH);
  
   string lstr(line);
  
   int fea_count = splitline(lstr, p_para->items, p_gbdt_model->info.fea_num+5, ' ');
    
   sscanf(p_para->items[0].c_str(),"%lf",&realv);
  
   for(int i = 2; i < fea_count; i++)
   {
   sscanf(p_para->items[i].c_str(),"%[^:]:%lf", feat, &value);
   int fid = atoi(feat);
   p_para->x_test[fid] = value;
   }
   fill_no_value_aver(p_gbdt_model, p_para->x_test);
  
   it = gbdt_regression_predict(p_gbdt_model, p_para->x_test, p_para->y_reg_predict);
  
   if(it < 0) 
   {
   fprintf(stderr, "predict error.\n");
   return NULL;
   }
  
   return p_para;//return para for simple process
  }
  
  /**
   * @brief the process result function
   * process the max entropy testing result here
   * */
  _SO_API_(int) process_result(void * arg, hmap *result,void * selfswitch)
  {
    if(!arg || !result) 
    {
      return -1;
    }
    
    stringstream ss;
    thread_para_t *p_para = (thread_para_t *)arg;
  
    ss << p_para->ori;
    (*result)["query"] = ss.str();
    ss.str("");
  
    ss << p_para->y_reg_predict;
    
    (*result)["y_predict"] = ss.str();
    ss.str("");
  
    return 0;
  }
  
