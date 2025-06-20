# **Design Discussion:**
I need your help in desining a solution for a data quality system. I need to take an input and generate an output. This output will be passed to a Large language model and we will use it to generate a summary of the data quality report and ask questions back and forth about the data quality. So after creating the output schema we need to transform it into a form which can be used by the LLM very efficiently. The output schema sample is given later in the text. Extend it to have more insights. I need to group the data quality results by source, tenant_id, dataset_uuid, dataset_name.

# **Input sample:**  
source: ABC System  
tenant_id: XYZC  
dataset_uuid: 2345sdfs  
dataset_name: EFGH_Dataset  
date: 2/24/2025  
dataset_record_count: 1591  
rule_code: 202  
level_of_execution: ATTRIBUTE  
attribute_name: status  
results: {"result": "Pass", "partialResult": false, "resultDescription": "Pass"}  
context_id: 2sdfg1232  
filtered_record_count: 1591  

Note: Input will be a csv file having 1 lakh rows. This script wil run on a consumer grade machine. So it should be optimized. Maybe you can use python pandas efficiently or let me know if any better tech is available which is suitable for this task.

## **Input Sample Schema:**
source: string  
tenant_id: string  
dataset_uuid: string  
dataset_name: string  
business_date: string   
dataset_record_count: number  
rule_code: number  
level_of_execution: string  
attribute_name: string  
results: json string  
context_id: string  
filtered_record_count: number  

Note: rule_code will be a number but there will be a mapping of rule_code to rule_name in the system. I will provide its json configuration below. This mapping has below sample values:

Rule ID: 202
Rule Name: ROW_COUNT
Rule Type: DATASET
Dimension: Correctness
Rule Description: Row count of the input file
Category: Category 1


# **Output sample:**
I need your suggestion here. I have below sample in mind. Extend it according to the use case.
 
source:    
tenant_id:    
dataset_uuid:    
dataset_name:    
level_of_execution:    
attribute_name: It should be a comma separted string having all the attribute names for which the rule has failed.    
rule_name:    
rule_type:    
dimension:    
rule_description:    
category_1_failure_count_total:    
category_2_failure_count_total:    
category_3_failure_count_total:    
category_4_failure_count_total:    
category_1_pass_count_total:  
category_2_pass_count_total:  
category_3_pass_count_total:  
category_4_pass_count_total:  
category_1_failure_count_1_month:    
category_2_failure_count_1_month:    
category_3_failure_count_1_month:    
category_4_failure_count_1_month:    
category_1_pass_count_1_month:    
category_2_pass_count_1_month:    
category_3_pass_count_1_month:    
category_4_pass_count_1_month:    
category_1_failure_count_3_month:    
category_2_failure_count_3_month:    
category_3_failure_count_3_month:    
category_4_failure_count_3_month:    
category_1_pass_count_3_month:    
category_2_pass_count_3_month:    
category_3_pass_count_3_month:    
category_4_pass_count_3_month:    
category_1_failure_count_1_year:    
category_2_failure_count_1_year:    
category_3_failure_count_1_year:    
category_4_failure_count_1_year:    
category_1_pass_count_1_year:    
category_2_pass_count_1_year:    
category_3_pass_count_1_year:    
category_4_pass_count_1_year:    


Note: You will need to use the results column in the input sample to calculate the count of pass and fail for each category. 
This is just a schema. I am not sure whether such a csv will be a good input for the LLM or should we transform it into a form which can be used by the LLM very efficiently like using all the above data and creating a plain text template and inject the values in that text template. You tell me what should be a good approach.


DO NOT WRITE ANY CODE. Just discuss with me the design. Once we both agree on the design you can create a markdown PRD and save it in the resources folder under prd.

Give me suggestions on how to have a more improved approach for the solution.